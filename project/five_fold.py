class FiveFold:

    # partition a data set into a list of 5 tuples for training and testing
    # five fold data partition
    def five_fold(data_set):
        partition_index = int( len(data_set) / 5 )
        print('pdex: ', partition_index) 
        s = 0
        fold = []
        for i in range(5): #0-4
            tr = data_set.copy()
            n = s + partition_index # was -1
            te = tr[s:n]
            del tr[s:s + partition_index]

            fold.append( (tr,te) )

            s += partition_index

        return fold