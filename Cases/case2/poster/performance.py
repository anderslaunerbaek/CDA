def performance(pred, Y):
    """
    asd
    """
    from sklearn.metrics import confusion_matrix
    import numpy as np

    def array_to_latex(tbl):
        for ii in range(tbl.shape[0]):
            tmp_str = ''
            for jj in range(tbl.shape[1]):
                if jj != 0:
                    tmp_str += ' & ' + "{:.0f}".format(tbl[ii,jj])  
                else:
                    tmp_str += "{:.0f}".format(tbl[ii,jj]) 

            tmp_str += ' \\\\ '
            print(tmp_str)

    def performance_measure(pred_test, Y_test):
        #
        cm = confusion_matrix(y_pred = pred_test,
            y_true = Y_test, 
            labels = list(range(len(set(Y_test)))))
        TP = np.diag(cm)
        FP = np.sum(cm, axis=0) - np.diag(cm)
        FN = np.sum(cm,axis=1) - np.diag(cm)
        TN = np.sum(cm) - (FP+FN+TP)
        #
        precision = TP/ (TP + FP)
        recall = TP / (TP + FN)
        F1 = np.multiply(2, np.multiply(precision, recall) / np.add(precision, recall))
        acc = (TP+TN)/(TP+FP+FN+TN)
        #
        return TP, FP, precision, recall, F1, acc, cm


    TP, FP, precision, recall, F1, Acc, cm = performance_measure(pred_test=pred, Y_test=np.argmax(Y, axis=1))
    print('--------------------------------------------')
    print('Average for all classes')
    print('Accurcy:   %f' %(np.mean(Acc)))
    print('Precision: %f' %(np.mean(precision)))
    print('Recall:    %f' %(np.mean(recall)))
    print('F1:        %f' %(np.mean(F1)))

    #
    print("std.\n")
    array_to_latex(cm)
    # 
    print("\npct.\n")
    cm_norm = cm / cm.astype(np.float).sum(axis=1, keepdims=True) * 100
    array_to_latex(cm_norm)

    print("\n\nPaste into latex..\n\n")
    tmp = np.ndarray((2,6))
    tmp[0:2,0:2] = cm_norm
    
    tmp[0:2,2] = precision * 100
    tmp[0:2,3] = recall * 100
    tmp[0:2,4] = F1 * 100
    tmp[0:2,5] = Acc * 100
    #
    array_to_latex(tmp)