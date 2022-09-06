import requests
import regex as re

BASE_URL = 'https://doaj.org/api/'


CHEMICAL_JOURNALS = [
    'The Journal of Biological Chemistry',
    'Chemical Science',
    'Frontiers in Chemistry',
    'Beilstein Journal of Organic Chemistry',
    'ACS Medicinal Chemistry Letters',
    'ChemistryOpen',
    'Chemistry Central Journal',
    'Journal of Enzyme Inhibition',
    'Medicinal Chemistry',
    'Journal of Analytical Methods in Chemistry',
    'The Journal of Automatic Chemistry',
    'Bioinorganic Chemistry',
    'International Journal of Analytical Chemistry',
    'Chemical Senses',
    'Future Medicinal Chemistry',
    'BMC Chemistry',
    'Food Chemistry: X'
]

JOINER = '+OR+'
PMID_FINDER = r'<Id>(\d+)</Id>'


if __name__ == '__main__':
    search_query = 'source=%7B"query"%3A%7B"bool"%3A%7B"must"%3A%5B%7B"terms"%3A%7B"index.schema_codes_tree.exact"%3A%5B"LCC%3AQD1-999"%5D%7D%7D%5D%7D%7D%2C"size"%3A"100"%2C"sort"%3A%5B%7B"created_date"%3A%7B"order"%3A"desc"%7D%7D%5D%2C"track_total_hits"%3Atrue%7D'
    
    endpoint = '/api/search/articles'    
    URL = 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=pmc&term='
    for idx, journal in enumerate(CHEMICAL_JOURNALS):
        URL += '"' + journal + '"[JOURNAL]'
        if idx < len(CHEMICAL_JOURNALS) - 1:
            URL += JOINER
    URL += '&RetMax=10000'

    queries = []
    for offset in range(0, 70000, 10000):
        suffix = f'&RetStart={offset}'
        queries.append(URL + suffix)
    
    pm_ids = []
    for query in queries:
        response = requests.get(query).content
        pm_ids += re.findall(PMID_FINDER, str(response))
    out_fn = 'pmid.txt'
    print(f'Saving {len(pm_ids)} files to {out_fn}')
    with open(out_fn, 'w') as fd:
        fd.write('\n'.join(pm_ids))
