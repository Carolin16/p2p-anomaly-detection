import os
import pandas as pd
import chromadb
from sentence_transformers import SentenceTransformer
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
class RAGExplainer:
    def __init__(self):
        # Load the embedding model (converts text to vectors)
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")

         # Set up ChromaDB (in-memory for now)
        self.chroma_client = chromadb.Client()
        self.cases_collection = self.chroma_client.create_collection(name="cases")
        self.docs_collection = self.chroma_client.create_collection(name="documents")
        
        
         # Set up OpenAI client
        self.openai_client = OpenAI(api_key = os.getenv("OPENAI_API_KEY"))
        
         # Load past cases into ChromaDB
        self.load_cases()

        self.load_documents()

    def load_cases(self):
        # Load the CSV
        df = pd.read_csv("data/p2p_invoices.csv")

        # Only keep rows that are anomalies
        anomalies = df[df["label"] == 1].head(200)

        past_data = []
        metadatas = []
        ids = []

        for i,row in anomalies.iterrows():


            # Convert the row into a human readable sentence
            text = (
                f"Anomaly type: {row['anomaly_type']}. "
                f"Vendor name: {row['vendor_name']}, "
                f"Vendor category: {row['vendor_category']}, " 
                f"Invoice amount: {row['invoice_amount']}, "
                f"Po amount: {row['po_amount']}, "
                f"Gr amount: {row['gr_amount']}, "
                f"Deviation: {row['deviation_pct']}%, "
                f"Days since last invoice: {row['days_since_last_invoice']}, "
                f"New vendor: {row['is_new_vendor']}, "
                f"Three-way match: {row['three_way_match']}."

            )

            past_data.append(text)
            metadatas.append({"anomaly_type" : row['anomaly_type'] , 'text' : text})
            ids.append(f"case_{i}")

        # Convert all documents to vectors in one go
        """
        Sentence-transformers converts all 200 sentences into 200 vectors of 384 numbers each
        """
        embeddings = self.embedder.encode(past_data).tolist()
        
        # Store everything in ChromaDB
        self.cases_collection.add(
            documents = past_data,
            embeddings=embeddings,
            metadatas=metadatas,
            ids = ids
        )

    def retrieve_similar(self,query_text:str):
        """
        The incoming invoice gets converted to 384 numbers
        Same process as when we loaded the past cases
        Compare it against the stored vectors
        """
        # Convert the incoming invoice text to a vector
        query_embedding = self.embedder.encode([query_text]).tolist()

        """
        ChromaDB compares the new invoice vector against all 200 stored vectors
        It calculates the cosine similarity for each one
        Returns the 3 closest matches

        ------Sample results------
            {
            
            "ids": [["case_5", "case_23", "case_87"]],
            "metadatas": [[
                    {"anomaly_type": "overbilling", "text": "Anomaly type: overbilling. Vendor: Vendor_42..."},
                    {"anomaly_type": "overbilling", "text": "Anomaly type: overbilling. Vendor: Vendor_11..."},
                    {"anomaly_type": "overbilling", "text": "Anomaly type: overbilling. Vendor: Vendor_67..."},
                ]],
            
            "distances": [[0.12, 0.18, 0.24]]  ← similarity scores
            }
        -----------------------------
        distances — how similar each case is (lower = more similar)
        """
        # Search ChromaDB for the 3 most similar past cases
        case_results = self.cases_collection.query(
            query_embeddings=query_embedding,
            n_results=3
        )
        docs_results = self.docs_collection.query(
            query_embeddings=query_embedding,
            n_results=2,
            where = {"doc_type" : "reference"}
        )

        """
        results comes back as a nested dictionary from ChromaDB
        We dig into metadatas to get the original human readable sentences back

        results = {
            "metadatas": [[...]],    # metadata if it was stored
            "documents": [[...]],    # the original text always
            "distances": [[...]]     # similarity scores
        }
        results["metadatas"][0] = [
            {"anomaly_type": "overbilling", "text": "Anomaly type..."},  # has metadata
            {},                                                           # empty - no metadata
            {}                                                            # empty - no metadata
        ]

        results["documents"] looks like this
        -------------------------------------
        results["documents"][0] = [
            "Anomaly type: overbilling. Vendor: GlobalParts Co...",   # from load_cases()
            "VENDOR FRAMEWORK CONTRACT — GlobalParts Co...",          # from load_documents()
            "DISPUTE LOG — INV-15450..."                              # from load_documents()
        ]
                """
        similar_cases = []
        for metadata in case_results["metadatas"][0]:
            #load cases
            similar_cases.append(metadata["text"])
        for docs in docs_results["documents"][0]:
            #load documents
            similar_cases.append(docs)

        
        print("Retrieved:")
        for case in similar_cases:
            print(case[:100])

        return similar_cases
    

    """
    ---------sample flag response from invoice api ---------
       flags = [
    {
        "anomaly_type": "overbilling",
        "is_anomaly": True,
        "reason": "Invoice amount exceeds PO amount"
    },
    {
        "anomaly_type": "duplicate_invoice",
        "is_anomaly": True,
        "reason": "Duplicate billing of invoice"
    }
    ]
    -----------------------------------------------------

    1. To build the query sentence — we take the first flag's anomaly type
    """
    def explain(self, invoice: dict, flags: list) -> str:
    
        # Collect all detected anomaly types
        all_types = ", ".join([f['anomaly_type'] for f in flags])
        # Step 1 - Build a sentence from the incoming invoice
        query_text = (
            f"Invoice ID: {invoice.get('invoice_id')}, " 
            f"PO reference: {invoice.get('po_reference')}, "
            f"Anomaly types: {all_types}. "
            f"Vendor: {invoice.get('vendor_name')}, "
            f"Vendor category: {invoice.get('vendor_category')}, "
            f"Invoice amount: {invoice.get('invoice_amount')}, "
            f"PO amount: {invoice.get('po_amount')}, "
            f"GR amount: {invoice.get('gr_amount')}, "
            f"Deviation: {invoice.get('deviation_pct')}%, "
            f"Days since last invoice: {invoice.get('days_since_last_invoice')}, "
            f"New vendor: {invoice.get('is_new_vendor')}, "
            f"Three way match: {invoice.get('three_way_match')}."

        )

        #retrieve top 3 similar cases
        similar_cases = self.retrieve_similar(query_text)

        """
        ---------sample----------------
        Case 1: Anomaly type: overbilling. Vendor: Vendor_42, Invoice amount: 8000...
        """
        cases_text = ""
        for i, case in enumerate(similar_cases) :
            
            case_number = i + 1
            line = f"Case {case_number}: {case}"
            cases_text = cases_text + "\n\n" + line
        

        """
        ------------sample ---------------

        overbilling: Invoice amount exceeds PO amount
        """
        flags_text = ""
        for flag in flags:
            
            line = f"- {flag['anomaly_type']}: {flag['reason']}"
            flags_text = flags_text + "\n" + line

        """
            generate prompt
        """

        prompt = f"""
        You are an AP fraud analyst reviewing a flagged vendor invoice.

        RESPONSE FORMAT — follow this exactly, no exceptions:

        FINDING: <one sentence only>

        EVIDENCE: <one sentence only>
         In the EVIDENCE section, 
         always write dollar amounts as plain text like $83,305.81 — never in backticks or code format

        RISK: <one sentence only>

        ACTION:
        - <step 1>
        - <step 2>

        STRICT RULES:
        - Each section starts on its own new line with its label
        - Never put EVIDENCE, RISK or ACTION content inside the FINDING section
        - Never wrap numbers in backticks — write $30,500 not `30,500`
        - Never use field names like days_since_last_invoice, gr_amount, po_amount — use plain English instead
        - Always cite invoice ID, vendor name, and dollar amounts
        - Keep total response under 100 words

        ---

        SIMILAR CONFIRMED FRAUD CASES:
        {cases_text}

        CURRENT INVOICE:
        {query_text}

        DETECTED ANOMALIES:
        {flags_text}

        Now write your analysis following the format above exactly.
        """
        response = self.openai_client.chat.completions.create(

            model = "gpt-4o",
            messages = [{"role":"user","content":prompt}]
            )

        return response.choices[0].message.content
    
    def load_documents(self , documents_dir = "documents"):

        texts = []
        ids = []
        metadatas_list = []
        for filename in os.listdir(documents_dir):
            filepath = os.path.join(documents_dir, filename)
        
            with open(filepath , "r") as f:
                text = f.read()
            
            texts.append(text)
            ids.append(f"doc_{filename}")

            if "policy" in filename or "contract" in filename :
                metadatas_list.append({"doc_type" : "reference"})
            else:
                metadatas_list.append({"doc_type" : "case_note"})
        #convert all 814 texts into vectors
        """
        [
            [0.23, -0.45, 0.81, ...],   # overbilling_policy.txt → 384 numbers
            [0.19, -0.41, 0.77, ...],   # contract_globalparts_co.txt → 384 numbers
            [0.21, -0.44, 0.79, ...],   # overbilling_INV-15450.txt → 384 numbers
            ...                          # 811 more
        ]
        """
        embeddings = self.embedder.encode(texts, batch_size=32,show_progress_bar=True ).tolist()

        self.docs_collection.add(
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas_list,
            ids = ids
        )
