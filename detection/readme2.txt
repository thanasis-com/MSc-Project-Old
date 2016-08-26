Function 'jaccardEvaluation' is used to calculate the jaccard index values.

Function 'falseRates' is used to calculate the false positives and false negatives per 100 nanodiscs. 
It also contains the method for how to plot 'False positives and false negatives per 100 nanodiscs' figure in our nanodisc detection and segmentation paper.

I mode a bit modifications in 'NewXMLReader' function, so please use this one to replace the previous one. The usage is still the same.

An example for how to use the two functions ('jaccardEvaluation' and 'falseRates') is in the 'evaluation' section in 'nanodiscAndGold.m' file.