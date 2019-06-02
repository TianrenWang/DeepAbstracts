from Bio import Entrez
import time
try:
    from urllib.error import HTTPError  # for Python 3
except ImportError:
    from urllib2 import HTTPError  # for Python 2

Entrez.email = "fwang100@uottawa.ca"
search_results = Entrez.read(Entrez.esearch(db="pubmed",
                                            term="p53",
                                            reldate=5000, datetype="pdat",
                                            usehistory="y"))
count = int(search_results["Count"])
print("Found %i results" % count)

maximum = 50000

batch_size = 1000
out_handle = open("p53-50000.txt", "w")
for start in range(0,maximum,batch_size):
    print("Downloading " + str(start) + " to " + str(start + batch_size))
    end = min(count, start+batch_size)
    print("Going to download record %i to %i" % (start+1, end))
    attempt = 1
    while attempt <= 3:
        print(attempt)
        try:
            fetch_handle = Entrez.efetch(db="pubmed",rettype="medline",
                                         retmode="text",retstart=start,
                                         retmax=batch_size,
                                         webenv=search_results["WebEnv"],
                                         query_key=search_results["QueryKey"])
            break
        except HTTPError as err:
            if 500 <= err.code <= 599:
                print("Received error from server %s" % err)
                print("Attempt %i of 3" % attempt)
                attempt += 1
                time.sleep(15)
            else:
                raise
    data = fetch_handle.read()

    #Write each abstract in the batch to the output file
    start = 0
    for i in range(batch_size):
        start = data.find("AB  ", start) + 6
        end = data.find(" - ", start) - 4
        abstract = data[start:end]
        abstract = abstract.replace('\n', ' ').replace('    ', ' ').replace('   ', '').replace('  ', ' ')
        start = end
        out_handle.write(abstract + '\n')

    fetch_handle.close()
out_handle.close()