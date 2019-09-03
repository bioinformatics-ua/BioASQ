

import pubmed_parser as pp

path = "..\\data\\medline17n0001.xml"

pubmed_dict = pp.parse_medline_xml(path)

print(pubmed_dict)