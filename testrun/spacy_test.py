from spacy.en import English
import time

now = time.time()
nlp = English()
print round(time.time()-now,3)

doc1 = "this's spacy tokenize test"
doc2 = "Darby Smart first launched a beta version of its website in 2013 and closed its Series A round in 2014 with $6.3 million."
doc3 = "'If you are selling a product for $12 and you see your competitor is selling it for $11.50 so you drop your price to $11.49 you have not accomplished anything except starting a price war,' he says."
doc4 = "ScoreBeyond, a service that helps students prepare for standardized tests like the SAT, said it has raised $2.8 million in a round led by Khosla Ventures."
doc5 = "Rami Eid is studying at Stony Brook University in New York"
doc6 = "Fitbit claims it was the biggest-selling activity tracker in the U.S. in the first quarter, according to a prospectus it filed with the Securities and Exchange Commission, and it filed for an IPO earlier this month to move in on the huge demand for the devices. However, a lawsuit might slow down Fitbit's ambitions and give its competitors an edge. Jawbone's UP wearable line"
doc7 = "[Update 3:30pm PST: Another source now tells us LiveRail sold for $500 million, matching the $400 million to $500 million range I reported earlier.]"
doc8 = "FirstBank's net income for full-year 2015 increased to $ 177.9 million from $175.4 million in 2014."
docs = [doc1, doc2, doc3, doc4, doc5, doc6, doc7, doc8]

for d in docs:
  f = unicode(d)
  o = nlp(f)
  for token in o:
    print token
  print "---------------"
