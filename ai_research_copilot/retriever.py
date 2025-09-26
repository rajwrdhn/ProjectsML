"""Module for retrieving data from a URL."""
import requests
import xml.etree.ElementTree as ET

def arxiv_retriever(query, max_result)->list:
    """Retrieve data from arXiv API."""
    url = f"http://export.arxiv.org/api/query?search_query=all:{query}+AND+cat:quant-ph&&start=0&max_results={max_result}"
    response = requests.get(url)
    if response.status_code != 200:
        raise Exception(f"Failed to retrieve data: {response.status_code}")
    
    root = ET.fromstring(response.content)
    papers = []
    for entry in root.findall('{http://www.w3.org/2005/Atom}entry'):
        title = entry.find('{http://www.w3.org/2005/Atom}title')
        abstract = entry.find('{http://www.w3.org/2005/Atom}summary')
        link = entry.find("{http://www.w3.org/2005/Atom}id")

        papers.append({
            "title": title.text.strip() if title is not None else "Untitled",
            "abstract": abstract.text.strip() if abstract is not None else "",
            "link": link.text if link is not None else "#"
        })
    
    return papers

def pubmed_retriever(retmax)->list:
    """Retrieve data from PubMed API."""
    url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=pubmed&term=electron&retmax={retmax}"
    response = requests.get(url)
    if response.status_code != 200:
        raise Exception(f"Failed to retrieve data: {response.status_code}")
    
    root = ET.fromstring(response.content)
    id_list = root.find('IdList')
    ids = [id_elem.text for id_elem in id_list.findall('Id')]
    
    summaries = []
    for pmid in ids:
        summary_url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi?db=pubmed&id={pmid}"
        summary_response = requests.get(summary_url)
        if summary_response.status_code != 200:
            continue
        summary_root = ET.fromstring(summary_response.content)
        docsum = summary_root.find('DocSum')
        title = docsum.find("Item[@Name='Title']").text
        summaries.append({'pmid': pmid, 'title': title})
    
    return summaries

if __name__ == "__main__":
    arxiv_data = arxiv_retriever(20)
    print("ArXiv Data:")
    for item in arxiv_data:
        print(f"Title: {item['title']}\nSummary: {item['summary']}\n")
    
    pubmed_data = pubmed_retriever(20)
    print("PubMed Data:")
    for item in pubmed_data:
        print(f"PMID: {item['pmid']}\nTitle: {item['title']}\n")

