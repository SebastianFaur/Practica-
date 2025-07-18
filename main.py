from langchain_community.document_loaders import PyPDFLoader
from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter 

loader = PyPDFLoader("SOAC_Matrix.pdf")
documents = loader.load()

if not documents:
    print("EROARE")
    exit() 

print(f"S-au încărcat {len(documents)} pagini din PDF.")
print("\nPrimele 500 de caractere din prima pagina:")
print(documents[0].page_content[:500])
print("...")
print(f"Metadate prima pagina: {documents[0].metadata}")
print("------------------------------\n")



text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=700,     
    chunk_overlap=150,   
    separators=["\n\n", "\n", " ", ""] 
)
docs = text_splitter.split_documents(documents)



print("Generare embeddings (acest pas poate dura)...")
embedding = OllamaEmbeddings(model="nomic-embed-text")
print("Embeddings generate.\n")


print("Creare vectorstore (FAISS) din embeddings...")
db = FAISS.from_documents(docs, embedding)
print("Vectorstore FAISS creat cu succes.\n")


print("Configurare LLM (mistral:instruct) și lant QA...")
llm = OllamaLLM(model="mistral:instruct")
retriever = db.as_retriever(search_kwargs={"k": 2})
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, return_source_documents=True)


while True:
    intrebare = input("\nIntrebare : ")
    if intrebare.lower() == "exit":
        print("La revedere!")
        break
    
    print("Cautare raspuns...")
    try:
        raspuns = qa_chain.invoke(intrebare)
        
        print("\n--- Raspuns: ---")
        print(raspuns["result"])
        
        if raspuns["source_documents"]:
            for i, doc in enumerate(raspuns["source_documents"]):
                print(f"Document {i+1} (Pagina: {doc.metadata.get('page', 'N/A')}):")
                print(f"{doc.page_content[:500]}...\n") 
        else:
            print("Nu s-au gasit documente bune.")
        print("-------------------------------------------\n")

    except Exception as e:
        print(f"A aparut o eroare la procesarea intrebarii: {e}")