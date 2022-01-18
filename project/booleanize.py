class Booleanize:

    # booleanize a data set given a threshold
    def booleanize(data_set, threshold):
        for row in data_set:
            for pos in range(1,len(row)):
                if row[pos] > threshold:
                    row[pos] = 1
                else:
                    row[pos] = 0
      
    # booleanize data set regarding if the feature occurs or not              
    def booleanize_occurances(data_set):
        for row in data_set:
            for pos in range(1,len(row)):
                if row[pos] != 0:
                    row[pos] = 1
                else:
                    row[pos] = 0
