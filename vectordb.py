from langchain_huggingface import HuggingFaceEmbeddings
# from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import Chroma
# import chromadb
from huggingface_hub import login
import torch
from sentence_transformers import SentenceTransformer
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import pipeline
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline 
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.chat_models import ChatOpenAI
import os, sys, argparse, time
import pandas as pd
from tqdm import tqdm

from dotenv import dotenv_values

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, choices=['c','train', 'test', 'inference'], default='c', help='Mode of the script')
    parser.add_argument('--train_file', type=str, default='train_100.csv', help='Name of the training file')
    parser.add_argument('--test_file', type=str, default='public_test.csv', help='Name of the test file')
    parser.add_argument('--corpus_file', type=str, default='corpus_100.csv', help='Name of the corpus file')
    parser.add_argument('--data_path', type=str, default='NaverLegal', help='Path to the output directory (preprocessed data)')
    parser.add_argument('--collection_name', type=str, default='DocsLegal3', help='Name of the collection')
    parser.add_argument('--vectordb', type=str, default='Docs_Legal3', help='Path to the vector database')
    parser.add_argument('--model_name', type=str, default='llama', help='model name to chat response')
    
    return parser.parse_args()

def create_vector_store(train_file, collection_name, vectordb):
    resource = train_file

    chunksize = 800000
    max_batch_size = 40000
    cnt = 0
    vectorstore = None
    init = 0
    for chunk in pd.read_csv(resource, chunksize=chunksize):
        cnt += 1
        documents = []
        ids = []
        metadatas = []
        for _, row in tqdm(chunk.iterrows(), total=chunk.shape[0], desc=f"Processing CSV {cnt}"):
            ct = row['context'][2: -3]
            if '"' in ct:
                ct = ct.replace('"', '')
            if r'\n' in ct:
                ct = ct.replace(r'\n', '')
            if r'/' in ct:
                ct = ct.replace(r'/', '')
            dic = f""" "Question": {row['question']}
                       "Context": {ct}"""
            documents.append(dic)
            ids.append(str(row['cid'][1:-1]))
            metadatas.append({"qid": row['qid']})
        print(documents[0], ids[0], metadatas[0])
        # q = r"""Người học ngành quản lý khai thác công trình thủy lợi trình độ cao đẳng phải có khả năng học tập và nâng cao trình độ như thế nào?"""
        # a = r"""Khả năng học tập, nâng cao trình độ\n- Khối lượng kiến thức tối thiểu, yêu cầu về năng lực mà người học phải đạt được sau khi tốt nghiệp ngành, nghề quản lý, khai thác các công trình thủy lợi, trình độ cao đẳng có thể tiếp tục phát triển ở các trình độ cao hơn;\n- Người học sau tốt nghiệp có năng lực tự học, tự cập nhật những tiến bộ khoa học công nghệ trong phạm vi ngành, nghề để nâng cao trình độ hoặc học liên thông lên trình độ cao hơn trong cùng ngành, nghề hoặc trong nhóm ngành, nghề hoặc trong cùng lĩnh vực đào tạo./."""
        # dic = f""" "Question": {q}
        #            "Context": {a}"""
        # documents.append(dic)
        # ids.append("1")
        # dicm = {"qid": "1"}
        # metadatas.append(dicm)
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model_kwargs = {'device': device, 'trust_remote_code': True}
        cache_dir = r"/teamspace/studios/this_studio/.cache"

        embeddings = HuggingFaceEmbeddings(
            model_name='dangvantuan/vietnamese-embedding-LongContext',
            cache_folder=cache_dir,
            model_kwargs=model_kwargs,
            show_progress=True,
        )
        if (len(documents) == len(ids)) and (len(ids) == len(metadatas)):
            print(f"Number of documents: {len(documents)}")
            time.sleep(1)

        if init == 0:
            print("Initializing ChromaDB")
            vectorstore = Chroma.from_texts(
                texts=documents,
                ids=ids,
                metadatas=metadatas,
                embedding=embeddings,
                persist_directory=vectordb,
                collection_name=collection_name,
            )
            init += 1
        else:
            vectorstore.add_texts(
                texts=documents,
                ids=ids,
                metadatas=metadatas,
                embedding=embeddings,
            )
    print(f"Collection {collection_name} created successfully in {vectordb}") 

def chatbot_response(content, collection_name, vectordb):
    args = parse_args()
    model_name = 'meta-llama/Llama-3.2-1B-Instruct' if args.model_name == 'llama' else args.model_name
    cache_dir = r"/teamspace/studios/this_studio/.cache"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        cache_dir=cache_dir,
        torch_dtype=torch.bfloat16,)

    pipe = pipeline(
        'text-generation',
        model=model,
        tokenizer=tokenizer,
        device_map="auto",
        max_new_tokens=600
        )
    llm = HuggingFacePipeline(
        pipeline=pipe,
        model_kwargs={'temperature': 0.6, 'top_p': 0.4},
    )

    model_kwargs = {'device': 'cpu'}
    embeddings = HuggingFaceEmbeddings(
        model_name='gbyuvd/ChemEmbed-v01',
        cache_folder='.cache',
        model_kwargs=model_kwargs,
        show_progress=True,
        )

    vectorstore = Chroma(
        embedding_function=embeddings,
        persist_directory=vectordb,
        collection_name=collection_name
    )
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    qa = ConversationalRetrievalChain.from_llm(llm=llm, 
                                               retriever=vectorstore.as_retriever(search_kwargs={'k':10}),
                                               verbose=False, memory=memory)
    chat = qa({'question': f'{content}'})
    print(chat['answer'])

def main():
    args = parse_args()
    input_path = args.input
    vectordb = args.vectordb
    input_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), input_path)
    vectordb = os.path.join(os.path.dirname(os.path.dirname(__file__)), vectordb)
    collection_name = args.collection_name
    case = args.case

    # chroma_path = Path(tempfile.gettempdir()) / 'vectorchroma_db'
    # vectordb = str(chroma_path)
    # client = chromadb.Client()
    # client = chromadb.PersistentClient(path=vectordb)
    # collection = client.get_or_create_collection(name=collection_name)
    login(token=venv["HF_TOKEN"], add_to_git_credential=True)
    match case:
        case 'createdb':
            create_vector_store(collection_name, vectordb)
        case 'qa':
            actIn = 'Acid mefenamic 500mg'
            content = f"""I am in need of converting the active ingredients from name to the molecular formula.
                                For example: ["Calcium gluconate"] -> C12H22CaO14
                                ["" warfarin "] -> C19H16O4
                                ["Tolvaptan"] -> C26H25ClN2O3
                                Sắt (III) hydroxyd polymaltose 34% -> C12H25FeO 
                                Desired Format: (molecular formula per line, all characters should be the same size, if there are many chemical formulas, write on a line that is separated by a comma (","). Do not write a lower number of atoms)
                                Please answer the whole chemical formula. No explanation. If you do not know or uncertain, print out "". Do not give words that are not related to chemical formulas, do not ask any more or suggestions.
                                Please convert the following active ingredient: {actIn}"""
            chatbot_response(content, collection_name, vectordb)

    # client.close()

if __name__ == '__main__':
    args = parse_args()

    mode = args.mode
    if mode == 'c':
        data_path = args.data_path
        data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), data_path)
        train_file = os.path.join(data_path, args.train_file)
        
        vectordb = args.vectordb
        vectordb = os.path.join(os.path.dirname(os.path.dirname(__file__)), vectordb)
        collection_name = args.collection_name
        # collection_name = "DocsLegal3"

        venv = dotenv_values(os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env"))
        login(token=venv["HF_TOKEN"], add_to_git_credential=True)
        create_vector_store(train_file, collection_name, vectordb)
    elif mode == 'test':
        venv = dotenv_values(os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env"))
        print(venv)


