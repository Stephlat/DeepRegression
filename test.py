import numpy as np
import cPickle as pickle
from sklearn.metrics import mean_absolute_error, explained_variance_score,mean_squared_error

def run_eval(Y_pred, Y_true, l, pbFlag,idOar="",printError=False):
    print "Evaluating"

    if (pbFlag == 'landmark'):
        # We need to change the shape of the Y_pred and Y_true matrices because the evaluation is different than in Biwi
        print Y_pred.shape
        Y_pred2 = np.reshape(Y_pred, (5*Y_pred.shape[0],2), order='C')
        Y_true2 = np.reshape(Y_true, (5*Y_true.shape[0],2), order='C')
        
        # mean squared error
        err = np.sqrt(np.sum((Y_pred2-Y_true2)**2, axis=1))

        listErr = np.empty((5,1))
        listFailures = np.empty((5,1))
        for i in range(5):
            temp = 0
            tempFailures = 0
            for j in xrange(i,len(err),5):
                temp += (err[j]/float(l))
                # If an error is larger than 5%, it is counted as failure.
                if (err[j]/float(l)) > 0.05:
                    tempFailures += 1
            listErr[i,0] = temp/(float(len(err))/5)
            listFailures[i,0] = tempFailures/(float(len(err))/5)

        print('Avg Detection Error:', listErr)
        print('Failure Rate:',listFailures)
        for x in listFailures:
            print " $" + str(100*x[0])+ "$ &"
        " ".join([str(100*x[0]) for x in listFailures])
        print np.mean(np.asarray([x[0] for x in listFailures]))
        
    elif (pbFlag == 'FBP'):
        LOGpred="/services/scratch/perception/slathuil/log/error_Training_"+str(idOar)+pbFlag+".pickle"
        pickle.dump((Y_pred,Y_true),open(LOGpred,"w+"))

        listSegments=[(0,1),(1,2),(3,4),(4,5),(6,7),(7,8),(9,10),(10,11),(12,13)]
        errX=np.empty((Y_pred.shape[0],14))
        errY=np.empty((Y_pred.shape[0],14))
        err=np.empty((Y_pred.shape[0],14))
        for j in range(14):
            errX[:,j]=Y_pred[:,2*j]-Y_true[:,2*j]
            errY[:,j]=Y_pred[:,2*j+1]-Y_true[:,2*j+1]

        # compute the error for the point in the middle of 3 and 4
        XYTorso_true=np.empty((Y_pred.shape[0],2))
        XYTorso_pred=np.empty((Y_pred.shape[0],2))
        XYTorso_true[:,0]=0.5*(Y_true[:,2*2]+Y_true[:,2*3])
        XYTorso_true[:,1]=0.5*(Y_true[:,2*2+1]+Y_true[:,2*3+1])
        XYTorso_pred[:,0]=0.5*(Y_pred[:,2*2]+Y_pred[:,2*3])
        XYTorso_pred[:,1]=0.5*(Y_pred[:,2*2+1]+Y_pred[:,2*3+1])
        errTorso=np.sqrt((XYTorso_pred[:,0]-XYTorso_true[:,0])**2+(XYTorso_pred[:,1]-XYTorso_true[:,1])**2,)


        
        err = np.sqrt((errX)**2+(errY)**2)
        
        lengthSegm=np.empty((Y_pred.shape[0],len(listSegments)))
        lengthTorso=np.empty(Y_pred.shape[0])

        for idSegm,seg in enumerate(listSegments):
            lengthSegm[:,idSegm]=np.sqrt((Y_true[:,seg[0]]-Y_true[:,seg[1]])**2)

        lengthTorso[:]=np.sqrt((XYTorso_true[:,0]-Y_true[:,2*12])**2+(XYTorso_true[:,1]-Y_true[:,2*12+1])**2)
            
        correct=np.empty((Y_pred.shape[0],len(listSegments)))
        correctTorso=np.empty(Y_pred.shape[0])
        for i in range(Y_pred.shape[0]):
            for idSegm,seg in enumerate(listSegments):
                if (err[i,seg[0]]/lengthSegm[i,idSegm])<0.5 and (err[i,seg[1]]/lengthSegm[i,idSegm])<0.5:
                    correct[i,idSegm]=1.0
                else:
                    correct[i,idSegm]=0.0
            if errTorso[i]/lengthTorso[i]<0.5 and err[i,12]/lengthTorso[i]<0.5:
                correctTorso[i]=1.0
            else:
                correctTorso[i]=0.0
                
        PCP=np.sum(correct,axis=0)/Y_pred.shape[0]
        PCPTorso=np.sum(correctTorso)/Y_pred.shape[0]
        print "head:   " + str(PCP[8])
        print "Torso:  " + str(PCPTorso)
        print "U Legs: " + str((PCP[1]+PCP[2])/2.0)
        print "L Legs: " + str((PCP[0]+PCP[3])/2.0)
        print "U Arms: " + str((PCP[5]+PCP[6])/2.0)
        print "L Arms: " + str((PCP[4]+PCP[7])/2.0)
        print "FB:     " + str((np.sum(PCP)+PCPTorso)/10.0)

    

    
    # mean absolute error
    MSE = mean_squared_error(Y_true, Y_pred, multioutput='raw_values')
    MAE = mean_absolute_error(Y_true, Y_pred, multioutput='raw_values')
    evs = explained_variance_score(Y_true, Y_pred, multioutput='raw_values')

    # Head pose estimation: pitch, yaw, roll
    print('Mean square error:', MSE,np.sum(MSE)/MSE.shape[0])
    print('Mean absolute error:', MAE,np.sum(MAE)/MAE.shape[0])
    print('Explained variances score:', evs)

