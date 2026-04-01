import json
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import io
from pypdf import PdfReader
import docx

from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

KEYWORDS1 = ('dokumenty', 'podania', 'zalaczniki', 'programy-studiow', 
             'efekty-uczenia-sie', 'karty-kursow', 'indywidualna-organizacja-studiow', 
             'do-pobrania', 'druki', 'regulamin-studiow', 'dokumenty-do-pobrania',
             'druki-i-formularze', 'pliki-do-pobrania', 'zarzadzenia-dziekana',
             'zaswiadczenia', 'wzory', 'karty-przedmitow', 'rozklady-zajec',
             'archiwalne-karty-przedmiotow', 'zasady-zaliczenia', 'regulamin',
             'regulamin-pracy')

def load_pdf_from_bytes(url):
    """Load pdf files from website and return Document object"""
    response = requests.get(url, timeout=5)
    response.raise_for_status()

    file_content = io.BytesIO(response.content)
    reader = PdfReader(file_content)

    docs = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        docs.append(Document(
            page_content=text,
            metadate={"source": url, "page": i+1}
        ))
    return docs

def load_docx_from_bytes(url):
    """Load docx files from website and return Document object"""
    response = requests.get(url, timeout=5)
    response.raise_for_status()

    file_content = io.BytesIO(response.content)

    docs = []
    document = docx.Document(file_content)
    full_text = [para.text for para in document.paragraphs]
    docs.append(Document(
        page_content="\n".join(full_text),
        metadata={"source": url, "format": "docs"}
    ))
    return docs

def embed_to_vector(url, file_type,
          split_chunk_size=1000,
          split_chunk_overlap=100,
          split_separators=["\n\n", "\n", " ", ""]):
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=split_chunk_size,
        chunk_overlap=split_chunk_overlap,
        separators=split_separators
    )

    try:
        if file_type == '.pdf':
            docs = load_pdf_from_bytes(url)
        elif file_type == '.docx':
            docs = load_docx_from_bytes(url)
        else:
            return []
        
        chunks = splitter.split_documents(docs)
        return chunks
    except Exception as e:
        return []



def crawler(base_url, 
            keywords = KEYWORDS1,
            embed = False,
            debug_mode = False):
    crawled = set()
    to_crawl = {base_url}
    files = set()

    if debug_mode: print(f'--- CRAWLING: {base_url} ---')

    # check all to_crawl urls
    while to_crawl:
        # take one url
        url = to_crawl.pop()
        if url in crawled:
            continue
    
        try:
            response = requests.get(url, timeout=5)
            # add to url to crawled
            crawled.add(url)

            bs = BeautifulSoup(response.text, 'html.parser')

            # look for links
            for link in bs.find_all('a', href=True):
                href = link['href']
                potential_url = urljoin(url, href)

                # ignore external url
                if urlparse(potential_url).netloc != urlparse(base_url).netloc:
                    continue

                # check extention
                extention = None
                if potential_url.lower().endswith('.pdf'): extention = '.pdf'
                elif potential_url.lower().endswith('.docx'): extention = '.docx'

                # look for files
                if extention and potential_url not in files:
                    files.add(potential_url)
                    if debug_mode: print(f'Found file: {potential_url}')
                    if embed: embed(potential_url, 'docx')
                    continue

                # look for new paths
                if potential_url not in crawled:
                    if any(word in potential_url.lower() for word in keywords):
                        to_crawl.add(potential_url)
        except Exception as e:
            pass



if __name__ == '__main__':
    crawler('https://wa.pwr.edu.pl/', embed=True, debug_mode=True)

    # with open('sources.json') as sf:
    #     sources = json.load(sf)['sources']

    # for source in sources:
    #     url = source['url']