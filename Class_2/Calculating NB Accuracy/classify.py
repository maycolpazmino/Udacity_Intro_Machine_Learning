def NBAccuracy(features_train, labels_train, features_test, labels_test):
    """ compute the accuracy of your Naive Bayes classifier """
    ### import the sklearn module for GaussianNB
    from sklearn.naive_bayes import GaussianNB

    ### create classifier
    clf = GaussianNB()

    ### fit the classifier on the training features and labels
    clf.fit(features_train, labels_train)

    ### use the trained classifier to predict labels for the test features
    pred = clf.predict(features_test)


    ### calculate and return the accuracy on the test data
    ### this is slightly different than the example, 
    ### where we just print the accuracy
    ### you might need to import an sklearn module
    
    ###### With the method .score of library sklearn.naive_bayes
    accuracy = clf.score(features_test, labels_test)
    
    ###### Import from sklearn.metrics the method accuracy_score
    from sklearn.metrics import accuracy_score
    accuracy2 =  accuracy_score(pred, labels_test)
    
    ###### Here you see 2 differents ways for accuracy method
    return accuracy, accuracy2
