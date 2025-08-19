import sys
import validators,streamlit as st
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import YoutubeLoader,UnstructuredURLLoader
from langchain_community.document_loaders import SeleniumURLLoader ,  WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter


## streamlit APP
st.set_page_config(page_title="LangChain: Summarize Text From YT or Website", page_icon="🦜")
st.title("🦜 LangChain: Summarize Text From YT or Website")
st.subheader('Summarize URL')


## Get the Groq API Key and url(YT or website)to be summarized
with st.sidebar:
    groq_api_key=st.text_input("Groq API Key",value="",type="password")

generic_url=st.text_input("URL",label_visibility="collapsed")

# first prompt (applied to each summaries)
map_prompt = PromptTemplate(
    template="""
    Summarize the following text clearly and concisely:
    {text}
    """,
    input_variables=["text"]
)

# Second prompt (applied to combined summaries)
combine_prompt = PromptTemplate(
    template="""
    Merge the following partial summaries into a single coherent summary:
    {text}
    """,
    input_variables=["text"]
)
if groq_api_key:
                if st.button("Summarize the Content from YT or Website"):
                    ## Validate all the inputs
                    if not groq_api_key.strip() or not generic_url.strip():
                        st.error("Please provide the information to get started")
                    elif not validators.url(generic_url):
                        st.error("Please enter a valid Url. It can may be a YT video utl or website url")
                
                    else:
                        try:
                            ## Gemma Model Using Groq API
                            llm =ChatGroq(model="gemma2-9b-it", groq_api_key=groq_api_key)
                            with st.spinner("Waiting..."):
                                ## loading the website or yt video data
                                if "youtube.com" in generic_url:
                                    loader=YoutubeLoader.from_youtube_url(generic_url,add_video_info=False)
                                else:
                                     loader = SeleniumURLLoader(urls=[generic_url])
                                     
                                docs=loader.load()
                                # smaller chunk 
                                text_splitter = RecursiveCharacterTextSplitter(
                                chunk_size=1000,  
                                chunk_overlap=100
                                 )
                                split_docs = text_splitter.split_documents(docs)
                                  
                                ## Chain For Summarization
                                chain=load_summarize_chain(llm,chain_type="map_reduce",map_prompt=map_prompt, combine_prompt=combine_prompt)
                                output_summary=chain.run(split_docs)
                
                                st.success(output_summary)
                        except Exception as e:
                            st.exception(f"Exception:{e}")
else:          st.info("Please Provide GROQ API KEY")

