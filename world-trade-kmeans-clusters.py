def Project_9_run_decision_tree_export_model(args):
    # ========

    def classify_trade_balance(df):
        """
        Classifies trade balance into 'Trade Surplus', 'Trade Deficit', or 'Balanced Trade'.
        """
        def balance_label(row):
            export = row['Export (US$ Thousand)']
            import_ = row['Import (US$ Thousand)']
            total_trade = export + import_
            balance = export - import_
            if balance > 0:
                return "Trade Surplus"
            elif balance < 0:
                return "Trade Deficit"
            else:
                if -0.8 * total_trade <= balance <= 0.8 * total_trade:
                    return "Balanced Trade"
                else:
                    return "Unknown"

        df['Trade Balance Category'] = df.apply(balance_label, axis=1)
        return df


    def classify_trade_intensity(df):
        """
        Categorizes trade intensity into 'Low Trade', 'Medium Trade', and 'High Trade'.
        """
        df['Total Trade'] = df['Export (US$ Thousand)'] + df['Import (US$ Thousand)']
        quantiles = df['Total Trade'].quantile([0.33, 0.66])

        def intensity_label(value):
            if value <= quantiles[0.33]:
                return "Low Trade"
            elif value <= quantiles[0.66]:
                return "Medium Trade"
            else:
                return "High Trade"

        df['Trade Intensity'] = df['Total Trade'].apply(intensity_label)
        return df.drop(columns=['Total Trade'])  # Dropping helper column


    def classify_tariff_impact(df):
        """
        Categorizes AHS Simple Average Tariff into 'Low Tariff', 'Moderate Tariff', and 'High Tariff'.
        """
        quantiles = df['AHS Simple Average (%)'].quantile([0.33, 0.66])

        def tariff_label(value):
            if value <= quantiles[0.33]:
                return "Low Tariff"
            elif value <= quantiles[0.66]:
                return "Moderate Tariff"
            else:
                return "High Tariff"

        df['Tariff Impact'] = df['AHS Simple Average (%)'].apply(tariff_label)
        return df


    def classify_export_dependence(df):
        """
        Categorizes export dependence as 'Export-Driven', 'Balanced', or 'Import-Driven'.
        """
        df['Export Share'] = df['Export (US$ Thousand)'] / (df['Export (US$ Thousand)'] + df['Import (US$ Thousand)'])

        def export_label(value):
            if value > 0.6:
                return "Export-Driven"
            elif value >= 0.4:
                return "Balanced"
            else:
                return "Import-Driven"

        df['Export Dependence'] = df['Export Share'].apply(export_label)
        return df.drop(columns=['Export Share'])  # Dropping helper column

    
    
    
    
    
    # =======
    # No, decision trees and train-test split are not the same, 
    # but they are related concepts in machine learning.

    # step 1. train test split
    # Module 8 Discussion
    # CS 332 Intro to Applied Data Science
    # Author: Tony Chan
    # Date: 2/26/2025
    # Purpose: To demonstrate the use of train_test_split function in sklearn.model_selection

    import pandas as pd
    # Show all columns
    pd.set_option('display.max_columns', 120)

    import seaborn as sns  # seaborn
    # Apply the default theme
    sns.set_theme()

    import matplotlib.pyplot as plt

    # 2. Read in data
    # clean, labeled, and quantitative data
    # filename = "C:/Users/Tony/OneDrive/Documents/CS332_Intro_to_Applied_Data_Science/Week_1/Module_1-Individual_Project_Assignment/world-trade-kmeans-clusters/mynew_proj7_cleanfile_clustered.csv"
    
    # filename = "C:/Users/Tony/OneDrive/Documents/CS332_Intro_to_Applied_Data_Science/Week_1/Module_1-Individual_Project_Assignment/world-trade-kmeans-clusters/tst1.csv"

    filename="C:/Users/Tony/OneDrive/Documents/CS332_Intro_to_Applied_Data_Science/Week_1/Module_1-Individual_Project_Assignment/world-trade-kmeans-clusters/MyNew_CleanFile_mod_7_proj_w_name.csv"
    print("Filename: ", filename)
    print("\n")

    #Read in the data
    MyData = pd.read_csv(filename)
    print("Read is successful\n")
    print("\n")

    # print(MyData)
    # print("\n")
    # print(MyData.info())
    # print("\n")

    # =======
    # 2. prepare/clean it for only quantitative data and it must be labeled
    # delete Company column
    MyData = MyData.drop(["Partner Name"], axis=1)
    print(MyData)
    print("\n")
    print(MyData.info())
    print("\n")

    # keep Label column in order to have training and testing balanced with respect to 
    # labels as required



    # # =======
    # # create target classifications in dataset
    # # 1. Trade Balance Category
    # MyData = classify_trade_balance(MyData)
    
    # # 2. Trade Intensity
    # MyData = classify_trade_intensity(MyData)
    
    # # 3. Tariff Impact
    # MyData = classify_tariff_impact(MyData)
    
    # 4. Export Dependence
    MyData = classify_export_dependence(MyData)

    print("create target classifications in dataset\n")
    print(MyData)
    print("\n")
    print(MyData.info())
    print("\n")
    


    # =======
    # Split the data into training and testing
    # train and test are balanced with respect to labels. Use stratify option
    from sklearn.model_selection import train_test_split




    # # --
    # # 1. Trade Balance Category
    # # Based on Net Trade Balance (Export - Import)
    # # Labels:
    # # "Trade Surplus" (Exports > Imports)
    # # "Trade Deficit" (Exports < Imports)
    # # "Balanced Trade" (Exports ≈ Imports)
    # train, test = train_test_split(MyData, test_size=0.3, stratify=MyData["Trade Balance Category"], random_state=42)
    
    # # 2. Trade Intensity
    # # Based on Total Trade Volume (Exports + Imports)
    # # Labels:
    # # "Low Trade" (Bottom 33% of total trade volume)
    # # "Medium Trade" (Middle 33%)
    # # "High Trade" (Top 33%)
    # train, test = train_test_split(MyData, test_size=0.3, stratify=MyData["Trade Intensity"], random_state=42)
    
    # # 3. Tariff Impact
    # # Tariff Impact Level
    # # Based on AHS Simple Average (%)
    # # Labels:
    # # "Low Tariff" (Bottom 33% of tariff values)
    # # "Moderate Tariff" (Middle 33%)
    # # "High Tariff" (Top 33%)
    # train, test = train_test_split(MyData, test_size=0.3, stratify=MyData["Tariff Impact"], random_state=42)
    
    # 4. Export Dependence
    # Based on Export Share (Export / (Export + Import))
    # Labels:
    # "Export-Driven" (Export Share > 60%)
    # "Balanced" (Export Share 40%-60%)
    # "Import-Driven" (Export Share < 40%)
    train, test = train_test_split(MyData, test_size=0.3, stratify=MyData["Export Dependence"], random_state=42)
    # --
    



    print("The Training Data is\n", train)
    print("\n")
    print("The Testing Data is\n", test)
    print("\n")



    # # =======
    # # (b) Visualize the Data
    # # think of three visualizations
    # # that will offer information about thedata. Each visualization should be different (so only one bar graph
    # # Do not describe the image - but rather explain the information that it shows.
    # # 1. A scatter plot of the data with the x-axis representing the year and the y-axis representing the export value.
    # # This visualization will show how the export value has changed over the years.
    # # 2. A scatter plot of the data with the x-axis representing the year and the y-axis representing the import value.
    # # This visualization will show how the import value has changed over the years.
    # # 3. A scatter plot of the data with the x-axis representing the export value and the y-axis representing the import value.
    # # This visualization will show the relationship between the export and import values.
    
    # # 1. A scatter plot of the data with the x-axis representing the year and the y-axis representing the export value.
    # plt.figure(figsize=(8, 6))
    # plt.scatter(MyData['Year'], MyData['Export (US$ Thousand)'])
    # plt.xlabel("Year")
    # plt.ylabel("Export (US$ Thousand)")
    # plt.title("Export Value Over the Years")
    # plt.show()
    
    # # 2. A scatter plot of the data with the x-axis representing the year and the y-axis representing the import value.
    # plt.figure(figsize=(8, 6))
    # plt.scatter(MyData['Year'], MyData['Import (US$ Thousand)'])
    # plt.xlabel("Year")
    # plt.ylabel("Import (US$ Thousand)")
    # plt.title("Import Value Over the Years")
    # plt.show()
    
    # # 3. A scatter plot of the data with the x-axis representing the export value and the y-axis representing the import value.
    # plt.figure(figsize=(8, 6))
    # plt.scatter(MyData['Export (US$ Thousand)'], MyData['Import (US$ Thousand)'])
    # plt.xlabel("Export (US$ Thousand)")
    # plt.ylabel("Import (US$ Thousand)")
    # plt.title("Export vs Import")
    # plt.show()
    
    # # 4. A scatter plot of the data with the x-axis representing the year and the y-axis representing the AHS Simple Average (%).
    # plt.figure(figsize=(8, 6))
    # plt.scatter(MyData['Year'], MyData['AHS Simple Average (%)'])
    # plt.xlabel("Year")
    # plt.ylabel("AHS Simple Average (%)")
    # plt.title("AHS Simple Average (%) Over the Years")
    # plt.show()
    

    # 5. export dependence column
    # 1. Export-Driven
    # 2. Balanced
    # 3. Import-Driven
    # 4. Unknown
    sns.countplot(x='Export Dependence', data=MyData)
    plt.title("Export Dependence Distribution")
    plt.show()



    # =======
    # 4.
    # after splitting the data, remove and retain the Training Label (from the Training Dataset)
    # remove and retain the Training Label (from the Training Dataset)
    # remove and retain the Testing Label (from the Testing Dataset)
    


    # # --
    # # 1. Trade Balance Category
    # train_label = train.pop("Trade Balance Category")
    # test_label = test.pop("Trade Balance Category")
    
    # # 2. Trade Intensity
    # train_label = train.pop("Trade Intensity")
    # test_label = test.pop("Trade Intensity")
    
    # # 3. Tariff Impact
    # train_label = train.pop("Tariff Impact")
    # test_label = test.pop("Tariff Impact")
    
    # 4. Export Dependence
    train_label = train.pop("Export Dependence")
    test_label = test.pop("Export Dependence")
    # --
    
    print("The Training Data is\n", train)
    print("\n")
    print("The Testing Data is\n", test)
    print("\n")

    print("The Training Label is\n", train_label)
    print("\n")
    print("The Testing Label is\n", test_label)
    print("\n")








    # =======
    # step 3. model training
    # from Mod_2_8_exer_decision_tree.py

    # Here is a step by step explanation of how decision trees work:
    # 1.	Dataset Preparation: You start with a dataset that you want to use for training 
    #       and evaluating your model.
    # 2.	Train-Test Split: You split this dataset into two subsets: 
    #           Training Set: Used to train the model. 
    #           Test Set: Used to evaluate the model's performance.
    # 3.	Model Training: You use the training set to train your decision tree model.
    # 4.	Model Evaluation: After training, you use the test set to evaluate how well 
    #       your model performs on unseen data.

    from sklearn.tree import DecisionTreeClassifier





    # # Remove the label from the dataset itself
    # # --
    # # 1. Trade Balance Category
    # MyData_noLabel = MyData.drop(["Trade Balance Category"], axis=1)
    
    # 2. Trade Intensity
    # MyData_noLabel = MyData.drop(["Trade Intensity"], axis=1)
    
    # # 3. Tariff Impact
    # MyData_noLabel = MyData.drop(["Tariff Impact"], axis=1)
    
    # 4. Export Dependence
    MyData_noLabel = MyData.drop(["Export Dependence"], axis=1)
    # --
    


    print(MyData_noLabel.head(15))
    print("MyData_noLabel length: ", len(MyData_noLabel))
    print("\n")
    
    print(train.head(15))
    print("train length: ", len(train))
    print("\n")
    
    print(train_label.head(15))
    print("train_label length: ", len(train_label))
    print("\n")
    





    # Step. Instantiate the decision tree using the defaults.
    # limit on the number of leaf nodes
    MyDT_Classifier = DecisionTreeClassifier(max_leaf_nodes=100, max_depth=5)
    # Use fit to create the decision tree model
    MyDT_Classifier = MyDT_Classifier.fit(train, train_label)
    #                                     ^^ data w/o test portion

    # after splitting the data into features and labels, we can now train the model.
    # The MyDT_Classifier.fit() method is used to train the decision tree model 
    # using the training dataset only. It takes two arguments: the features from 
    # the training dataset (train) and the corresponding labels (train_label).
    
    # how to print the tree
    print(MyDT_Classifier)
    print("\n")

    # =======
    # Step Visualize the decision tree 
    # use matplotlib from module 9 exercise decision tree classifier
    import matplotlib.pyplot as plt
    from sklearn import tree

    # Step 3: Visualize the Decision Tree
    print(MyDT_Classifier.classes_)
    print(train.columns)
    ##Tree Plot Option 1
    MyPlot=tree.plot_tree(MyDT_Classifier,
                       feature_names=train.columns, 
                       class_names=MyDT_Classifier.classes_,
                       filled=True)

    ## !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    ## To see the tree, open this file on your computer :)
    ## !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    # plt.savefig("MyTree.jpg")    
    plt.savefig("MyTree.png", dpi = 720)
    plt.close()




    # =======
    import seaborn as sns
    from sklearn.metrics import confusion_matrix
    # from sklearn.metrics import ConfusionMatrixDisplay

    # Step 4: Use the Decision Tree Model to Predict the Test Data and Compare with the Actual Labels
    ## Predict the Testing Dataset
    Prediction=MyDT_Classifier.predict(test)
    print("The Prediction is\n", Prediction)
    print(Prediction)
    print("\n")



    # # --
    # # 1. Trade Balance Category
    # label_names= ['Trade Surplus', 'Trade Deficit', 'Balanced Trade']
    
    # # 2. Trade Intensity
    # label_names= ['Low Trade', 'Medium Trade', 'High Trade']
    
    # # 3. Tariff Impact
    # label_names= ['Low Tariff', 'Moderate Tariff', 'High Tariff']
    
    # 4. Export Dependence
    label_names= ['Export-Driven', 'Balanced', 'Import-Driven']
    # --
    


    Actual_Labels=test_label
    Predicted_Labels=Prediction

    ##Create the Basic Confusion Matrix as a heatmap in Seaborn.
    ## Note that you can also use Sklearn's ConfusionMatrixDisplay
    My_Conf_Mat = confusion_matrix(Actual_Labels, Predicted_Labels)
    print("The Confusion Matrix is\n", My_Conf_Mat)
    print(My_Conf_Mat)
    print("\n")




    # # Create the fancy CM using Seaborn
    # # --
    # # # 1. Trade Balance Category
    # sns.heatmap(My_Conf_Mat, annot=True,cmap='Blues',xticklabels=label_names, yticklabels=label_names, cbar=False)
    # plt.title("Confusion Matrix For Trade Balance Data",fontsize=20)
    # plt.xlabel("Actual", fontsize=15)
    # plt.ylabel("Predicted", fontsize=15)
    # plt.show()
    
    # # 2. Trade Intensity
    # sns.heatmap(My_Conf_Mat, annot=True,cmap='Blues',xticklabels=label_names, yticklabels=label_names, cbar=False)
    # plt.title("Confusion Matrix For Trade Intensity Data",fontsize=20)
    # plt.xlabel("Actual", fontsize=15)
    # plt.ylabel("Predicted", fontsize=15)
    # plt.show()
    
    # # 3. Tariff Impact
    # sns.heatmap(My_Conf_Mat, annot=True,cmap='Blues',xticklabels=label_names, yticklabels=label_names, cbar=False)
    # plt.title("Confusion Matrix For Tariff Impact Data",fontsize=20)
    # plt.xlabel("Actual", fontsize=15)
    # plt.ylabel("Predicted", fontsize=15)
    # plt.show()
    
    # 4. Export Dependence
    sns.heatmap(My_Conf_Mat, annot=True,cmap='Blues',xticklabels=label_names, yticklabels=label_names, cbar=False)
    plt.title("Confusion Matrix For Export Dependence Data",fontsize=20)
    plt.xlabel("Actual", fontsize=15)
    plt.ylabel("Predicted", fontsize=15)
    plt.show()
    # --




    
    # =======
    # import graphviz  # If this does not work for you, just comment it out for now.
    # import export_graphviz
    

    # !!!!!!!!!!!!!!!!!
    # To run this next part, you will need to 
    # conda install python-graphviz
    # pip install graphviz
    # If you cannot get it to work - don't worry about that for now
    # Comment out the following lines.

    # TREE_Vis = export_graphviz(MyDT_Classifier, 
    #                                 out_file=None, 
    #                                 feature_names=train.columns,  
    #                                 class_names=["Risk", "NoRisk", "Medium"],  
    #                                 filled=True, rounded=True,  
    #                                 special_characters=True)  

    # graph = graphviz.Source(TREE_Vis)  
    # graph

    # use export_graphviz() to create a dot file to run in graphviz standalone product
    # no TREE_Vis
    # =======


    






def Project_7_run_kmeans_trade_clustering(args):
    # Module 7 Project
    # CS332 Winter 2025
    # 2/21/25
    # Tony Chan

    # See Module 2 and Module 6 Quix for more code on Titanic
    # From Module 7 Exercise

    import seaborn as sns  # seaborn
    # Apply the default theme
    sns.set_theme()


    import pandas as pd
    import matplotlib.pyplot as plt
    # Show all columns
    pd.set_option('display.max_columns', 120)



    # =======
    # # This is the path on local machine to data file
    # dirty
    # filename="C:/Users/Tony/OneDrive/Documents/CS332_Intro_to_Applied_Data_Science/Week_1/Module_1-Individual_Project_Assignment/kaggle_world_import_export/MyNew_CleanFile_mod_7_proj_w_name.csv"


    # 5)
    # clean and yes Survived. yes Sib_Parch
    filename="C:/Users/Tony/OneDrive/Documents/CS332_Intro_to_Applied_Data_Science/Week_1/Module_1-Individual_Project_Assignment/kaggle_world_import_export/MyNew_CleanFile_mod_7_proj_w_name.csv"
    print("Filename: ", filename)
    print("\n")

    # --
    # question 1

    MyData = pd.read_csv(filename)
    print("Data loaded successfully")
    print("\n")











    # # --
    # # Take the "Partner Name" column off the dataframe and save it as a LIST.

    # Before
    # print(type(MyData))       # Print the type of MyData
    # print(MyData)             # print the data frame
    print(MyData.head(20))
    print("\n")
    # print(MyData.head())
    # print("\n")
    # print(MyData.describe())  # summary statistics on numeric columns
    # print("\n")
    # print(MyData.isnull().sum())   # how many nulls per column
    # print("\n")
    print(MyData.info())
    print("\n")
    # print(MyData.dtypes)    # Print the data types of each column in MyData
    # print("\n")
    # # print all columns
    # print(MyData.columns)   # print all columns
    # print("\n")








    # Take the "Partner Name" column off the dataframe and save it as a LIST.
    Partner = MyData['Partner Name'].tolist()
    # print(Partner)
    # print("\n")
    print("Partner Name column has been saved as a list")
    print("\n")



    # Remove the "Partner Name" column from the dataframe.
    MyData.drop(columns=['Partner Name'], inplace=True)

    # Remove the "AHS Simple Average (%)" column from the dataframe.
    MyData.drop(columns=['AHS Simple Average (%)'], inplace=True)
    



    # # After
    # # print(type(MyData))       # Print the type of MyData
    # # print(MyData)             # print the data frame
    # print(MyData.head(20))
    # print("\n")
    # # print(MyData.head())
    # # print("\n")
    # # print(MyData.describe())  # summary statistics on numeric columns
    # # print("\n")
    # # print(MyData.isnull().sum())   # how many nulls per column
    # # print("\n")
    # print(MyData.info())
    # print("\n")
    # # print(MyData.dtypes)    # Print the data types of each column in MyData
    # # print("\n")
    # # # print all columns
    # # print(MyData.columns)   # print all columns
    # # print("\n")


    # # # save the cleaned dataframe to a new csv file
    # # MyData.to_csv('MyNew_CleanFile_Survived.csv', index=False)
    # # print("Data saved to MyNew_CleanFile_Survived.csv")






    # =======
    # Part 2
    # Module 7
    # KMeans using sklearn
    # see module 2


    # 2)


    # Before
    # print(type(MyData))       # Print the type of MyData
    # print(MyData)             # print the data frame
    # print(MyData.head(20))
    # print("\n")
    # print(MyData.head())
    # print("\n")
    # print(MyData.describe())  # summary statistics on numeric columns
    # print("\n")
    # print(MyData.isnull().sum())   # how many nulls per column
    # print("\n")
    # print(MyData.info())
    # print("\n")
    # print(MyData.dtypes)    # Print the data types of each column in MyData
    # print("\n")
    # # print all columns
    # print(MyData.columns)   # print all columns
    # print("\n")


    # # --
    # # do kmeans clustering from skit-learn
    # from sklearn.cluster import KMeans
    # from sklearn.preprocessing import StandardScaler
    
        
    # # Create a StandardScaler object
    # scaler = StandardScaler()
    
    # # Fit the scaler to the data
    # scaler.fit(MyData)
    
    # # Transform the data using the fitted scaler
    # X = scaler.transform(MyData)
    # # --
    


    # # c) how to determine best k value
    # # Calculate WCSS for a range of k values
    # wcss = []
    # for i in range(1, 11):
    #     kmeans = KMeans(n_clusters=i, random_state=0)
    #     kmeans.fit(X)
    #     wcss.append(kmeans.inertia_)

    # # Plot the Elbow Curve
    # plt.plot(range(1, 11), wcss, marker='o')
    # plt.title('Elbow Method')
    # plt.xlabel('Number of clusters')
    # plt.ylabel('WCSS')
    # plt.show()





    # # --
    # # 2) Create a KMeans model with 4 clusters
    # k = 4
    # kmeans = KMeans(n_clusters=k, random_state=0)

    # # Fit the model to the data
    # kmeans.fit(X)

    # # Get the cluster labels
    # labels = kmeans.labels_

    # # Add the cluster labels to the dataframe
    # MyData['Cluster'] = labels

    # # Print the cluster labels
    # print(labels)
    # print("\n")

    # # Assuming kmeans is your KMeans model
    # centroids = kmeans.cluster_centers_
    # print("Cluster centroids:\n", centroids)
    # print("\n")

    # # Group by cluster and calculate summary statistics
    # cluster_summary = MyData.groupby('Cluster').mean()
    # print("Cluster summary:\n", cluster_summary)
    # print("\n")

    # # print clusters
    # print(MyData.head(20))
    # print("\n")
    # # --





    import numpy as np
    from sklearn.cluster import KMeans

    # Example data from k=3 clustering
    data = np.array([
    [2009.319149,           5.988350e+09,           5.855941e+09],
    [2013.500000,           1.872798e+10,           1.686070e+10],
    [1996.680194,           3.815408e+07,           3.692280e+07],
    [2013.461970,           1.052814e+08,           9.182374e+07]
    ])

    # Fit the K-Means model
    kmeans = KMeans(n_clusters=4, random_state=0).fit(data)

    # Function to generate a new data vector for a given cluster
    def generate_vector_for_cluster(cluster_number):
        # Get the centroid of the cluster
        centroid = kmeans.cluster_centers_[cluster_number]
    
        # Get the points in the cluster
        cluster_points = data[kmeans.labels_ == cluster_number]
    
        # Calculate the standard deviation of the points in the cluster
        std_dev = np.std(cluster_points, axis=0)
    
        # Generate a new data vector around the centroid using the standard deviation
        new_data_vector = np.random.normal(centroid, std_dev)
    
        return new_data_vector

    # Example usage 0,1,2,3
    cluster_number = 0
    new_data_vector = generate_vector_for_cluster(cluster_number)

    print(f'Generated new data vector for cluster {cluster_number}: {new_data_vector}')
    print("\n")













    # # b. Create a scatter plot of the dataset with the points colored 
    # # by the cluster labels. use dark color for points
    # # 2D scatter plot
    # plt.scatter(MyData['Export (US$ Thousand)'], MyData['Import (US$ Thousand)'], c=MyData['Cluster'], cmap='Dark2')
    # plt.xscale('log')
    # plt.yscale('log')
    # plt.xlabel('Export (US$ Thousand)')
    # plt.ylabel('Import (US$ Thousand)')
    # plt.title('Export vs Import')
    # plt.show()
    

    # # b. Create a 3D scatter plot using Year, Export (US$ Thousand), Import (US$ Thousand) with the points contrasting colored
    # # by the cluster labels. use dark color for points
    # from mpl_toolkits.mplot3d import Axes3D
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.scatter(MyData['Year'], MyData['Export (US$ Thousand)'], MyData['Import (US$ Thousand)'], c=MyData['Cluster'], cmap='Dark2')
    # ax.set_xlabel('Year')
    # ax.set_ylabel('Export (US$ Thousand)')
    # ax.set_zlabel('Import (US$ Thousand)')
    # plt.title('Year vs Export vs Import')
    # plt.show()
    
    
    






    # # Add the 'Partner Name' list back to the DataFrame
    # # Survived was saved to variable Survived
    # MyData['Survived'] = Survived

    # # print(MyData.head(20))

    # # Analyze survival distribution within clusters
    # survival_distribution = MyData.groupby('Cluster')['Survived'].value_counts().unstack()
    # print("Survival distribution within clusters:\n")
    # print(survival_distribution)

    # # Optionally, visualize the survival distribution within clusters
    # survival_distribution.plot(kind='bar', stacked=True)
    # plt.title('Survival Distribution within Clusters')
    # plt.xlabel('Cluster')
    # plt.ylabel('Count')
    # plt.xticks(rotation=0)
    # plt.show()




    # # 22. save the dataframe with the cluster labels to a new csv file
    # MyData.to_csv('mynew_proj7_cleanfile_clustered.csv', index=False)
    # print("Data saved to mynew_proj7_cleanfile_clustered.csv")







def Project_7_clean_kaggle_world_trade_data_kaggle_world_trade_data(args):
    # Module 7 Project
    # CS332 Winter 2025
    # 2/21/25
    # Tony Chan

    # See Module 2 and Module 6 Quix for more code on Titanic
    # From Module 7 Exercise

    import seaborn as sns  # seaborn
    # Apply the default theme
    sns.set_theme()


    import pandas as pd
    import matplotlib.pyplot as plt
    # Show all columns
    pd.set_option('display.max_columns', 120)



    # =======
    # # This is the path on local machine to data file
    # dirty
    filename="C:/Users/Tony/OneDrive/Documents/CS332_Intro_to_Applied_Data_Science/Week_1/Module_1-Individual_Project_Assignment/kaggle_world_import_export/34_years_world_export_import_dataset.csv"


    # --
    # filename=""
    print("Filename: ", filename)
    print("\n")


    # --
    MyData = pd.read_csv(filename)
    print("Data loaded successfully")
    print("\n")




    # --
    # visualize the data

    # # 1.
    # # histogram of column "Export (US$ Thousand)"
    # plt.hist(MyData['Export (US$ Thousand)'])
    # plt.xlabel('Export (US$ Thousand)')
    # plt.ylabel('Frequency')
    # plt.title('Histogram of Export (US$ Thousand)')
    # plt.yscale('log')  # Set y-axis to log scale
    # plt.show()
    

    # # 2.
    # # histogram of column "Import (US$ Thousand)"
    # plt.hist(MyData['Import (US$ Thousand)'])
    # plt.xlabel('Import (US$ Thousand)')
    # plt.ylabel('Frequency')
    # plt.title('Histogram of Import (US$ Thousand)')
    # plt.yscale('log')  # Set y-axis to log scale
    # plt.show()
    

    # # 3.
    # # histogram of column "AHS Simple Average (%)"
    # plt.hist(MyData['AHS Simple Average (%)'])
    # plt.xlabel('AHS Simple Average (%)')
    # plt.ylabel('Frequency')
    # plt.title('Histogram of AHS Simple Average (%)')
    # # plt.yscale('log')  # Set y-axis to log scale
    # plt.show()
    
    # # 4.
    # # histogram of column "Year"
    # plt.hist(MyData['Year'])
    # plt.xlabel('Year')
    # plt.ylabel('Frequency')
    # plt.title('Histogram of Year')
    # # plt.yscale('log')  # Set y-axis to log scale
    # plt.show()
    


    # # 5.
    # # line plot of column "Year" versus column "Export (US$ Thousand)"
    # plt.plot(MyData['Year'], MyData['Export (US$ Thousand)'])   
    # plt.xlabel('Year')
    # plt.ylabel('Export (US$ Thousand)')
    # plt.title('Year vs Export (US$ Thousand)')
    # plt.show()
    

    # # 6.
    # # line plot of column "Year" versus column "Import (US$ Thousand)"
    # plt.plot(MyData['Year'], MyData['Import (US$ Thousand)'])
    # plt.xlabel('Year')
    # plt.ylabel('Import (US$ Thousand)')
    # plt.title('Year vs Import (US$ Thousand)')
    # plt.show()
    
    # # 7.
    # # line plot of column "Year" versus column "AHS Simple Average (%)"
    # plt.plot(MyData['Year'], MyData['AHS Simple Average (%)'])
    # plt.xlabel('Year')
    # plt.ylabel('AHS Simple Average (%)')
    # plt.title('Year vs AHS Simple Average (%)')
    # plt.show()
    


    # --
    # b. Delete columns

    # # Before
    # # print(type(MyData))       # Print the type of MyData
    # # print(MyData)             # print the data frame
    # print(MyData.head(15))
    # print("\n")
    # # print(MyData.head())
    # # print("\n")
    # # print(MyData.describe())  # summary statistics on numeric columns
    # # print("\n")
    # # print(MyData.isnull().sum())   # how many nulls per column
    # # print("\n")
    # print(MyData.info())
    # print("\n")
    # # print(MyData.dtypes)    # Print the data types of each column in MyData
    # # print("\n")
    # # # print all columns
    # # print(MyData.columns)   # print all columns
    # # print("\n")


    # delete columns 4 though 9
    MyData = MyData.drop(MyData.columns[4:9], axis=1)
    # delete columns 5 through last column
    MyData = MyData.drop(MyData.columns[5:], axis=1)

    # # After
    # # print(type(MyData))       # Print the type of MyData
    # # print(MyData)             # print the data frame
    # print(MyData.head(15))
    # print("\n")
    # # print(MyData.head())
    # # print("\n")
    # # print(MyData.describe())  # summary statistics on numeric columns
    # # print("\n")
    # # print(MyData.isnull().sum())   # how many nulls per column
    # # print("\n")
    # print(MyData.info())
    # print("\n")
    # # print(MyData.dtypes)    # Print the data types of each column in MyData
    # # print("\n")
    # # # print all columns
    # # print(MyData.columns)   # print all columns
    # # print("\n")







    # # =======
    # # --
    # # replace missing values

    # # Before
    # # print(type(MyData))       # Print the type of MyData
    # # print(MyData)             # print the data frame
    # print(MyData.head(20))
    # print("\n")
    # # print(MyData.head())
    # # print("\n")
    # # print(MyData.describe())  # summary statistics on numeric columns
    # # print("\n")
    # # print(MyData.isnull().sum())   # how many nulls per column
    # # print("\n")
    # print(MyData.info())
    # print("\n")
    # # print(MyData.dtypes)    # Print the data types of each column in MyData
    # # print("\n")
    # # # print all columns
    # # print(MyData.columns)   # print all columns
    # # print("\n")



    # replace missing values, NaN, Null values in "Partner Name" with "Unknown"
    MyData['Partner Name'] = MyData['Partner Name'].fillna('Unknown')
    
    # replace missing values, NaN, Null values in "Year" with "0"
    MyData['Year'] = MyData['Year'].fillna(0)
    
    # replace missing values, NaN, Null values in "Export (US$ Thousand)" with mean of column
    # and same "Partner Name" in row
    MyData['Export (US$ Thousand)'] = MyData['Export (US$ Thousand)'].fillna(MyData.groupby('Partner Name')['Export (US$ Thousand)'].transform('mean'))
    
    # replace missing values, NaN, Null values in "Import (US$ Thousand)" with mean of column
    # and same "Partner Name" in row
    MyData['Import (US$ Thousand)'] = MyData['Import (US$ Thousand)'].fillna(MyData.groupby('Partner Name')['Import (US$ Thousand)'].transform('mean'))
    
    # replace missing values, NaN, Null values in "AHS Simple Average (%)" with mean of column
    # and same "Partner Name" in row
    MyData['AHS Simple Average (%)'] = MyData['AHS Simple Average (%)'].fillna(MyData.groupby('Partner Name')['AHS Simple Average (%)'].transform('mean'))
    




    # # After
    # # print(type(MyData))       # Print the type of MyData
    # # print(MyData)             # print the data frame
    # print(MyData.head(20))
    # print("\n")
    # # print(MyData.head())
    # # print("\n")
    # # print(MyData.describe())  # summary statistics on numeric columns
    # # print("\n")
    # # print(MyData.isnull().sum())   # how many nulls per column
    # # print("\n")
    # print(MyData.info())
    # print("\n")
    # # print(MyData.dtypes)    # Print the data types of each column in MyData
    # # print("\n")
    # # # print all columns
    # # print(MyData.columns)   # print all columns
    # # print("\n")
















    # # --
    # # correct for wrong values


    # # Before
    # # print(type(MyData))       # Print the type of MyData
    # # print(MyData)             # print the data frame
    # print(MyData.head(20))
    # print("\n")
    # # print(MyData.head())
    # # print("\n")
    # # print(MyData.describe())  # summary statistics on numeric columns
    # # print("\n")
    # # print(MyData.isnull().sum())   # how many nulls per column
    # # print("\n")
    # print(MyData.info())
    # print("\n")
    # # print(MyData.dtypes)    # Print the data types of each column in MyData
    # # print("\n")
    # # # print all columns
    # # print(MyData.columns)   # print all columns
    # # print("\n")

    # find anomalies values in "Partner Name" column
    print(MyData['Partner Name'].unique())
    print("\n")


    # find non number values in "Year" column
    print(MyData['Year'].unique())
    print("\n")
    
    # find anomalies values in "Export (US$ Thousand)" column
    print(MyData['Export (US$ Thousand)'].unique())
    print("\n")
    
    # find anomalies values in "Import (US$ Thousand)" column
    print(MyData['Import (US$ Thousand)'].unique())
    print("\n")
    
    # find anomalies values in "AHS Simple Average (%)" column
    print(MyData['AHS Simple Average (%)'].unique())
    print("\n")
    





    # # --
    
    # do visualizations in order to find anomalies values in columns first
    # then correct them here
    
    # # correct for values that are out of range, less than zero or zero
    # make all values in "Year" column that are less than 1950 to 0
    MyData.loc[MyData['Year'] < 1950, 'Year'] = 0
    
    # make all values in "Export (US$ Thousand)" column that are less than 0 to mean of column
    # and same "Partner Name" in row
    MyData.loc[MyData['Export (US$ Thousand)'] < 0, 'Export (US$ Thousand)'] = MyData.groupby('Partner Name')['Export (US$ Thousand)'].transform('mean')
    
    # make all values in "Import (US$ Thousand)" column that are less than 0 to mean of column
    # and same "Partner Name" in row
    MyData.loc[MyData['Import (US$ Thousand)'] < 0, 'Import (US$ Thousand)'] = MyData.groupby('Partner Name')['Import (US$ Thousand)'].transform('mean')
    
    # make all values in "AHS Simple Average (%)" column that are less than 0 to mean of column
    # and same "Partner Name" in row
    MyData.loc[MyData['AHS Simple Average (%)'] < 0, 'AHS Simple Average (%)'] = MyData.groupby('Partner Name')['AHS Simple Average (%)'].transform('mean')
    

    




    # After
    # print(type(MyData))       # Print the type of MyData
    # print(MyData)             # print the data frame
    print(MyData.head(20))
    print("\n")
    # print(MyData.head())
    # print("\n")
    # print(MyData.describe())  # summary statistics on numeric columns
    # print("\n")
    # print(MyData.isnull().sum())   # how many nulls per column
    # print("\n")
    print(MyData.info())
    print("\n")
    # print(MyData.dtypes)    # Print the data types of each column in MyData
    # print("\n")
    # # print all columns
    # print(MyData.columns)   # print all columns
    # print("\n")


    # find anomalies values in "Partner Name" column
    print(MyData['Partner Name'].unique())
    print("\n")


    # find non number values in "Year" column
    print(MyData['Year'].unique())
    print("\n")
    
    # find anomalies values in "Export (US$ Thousand)" column
    print(MyData['Export (US$ Thousand)'].unique())
    print("\n")
    
    # find anomalies values in "Import (US$ Thousand)" column
    print(MyData['Import (US$ Thousand)'].unique())
    print("\n")
    
    # find anomalies values in "AHS Simple Average (%)" column
    print(MyData['AHS Simple Average (%)'].unique())
    print("\n")







    # # --
    # # update datatypes
    # Once there are no null values or incorrect datatyepes then datatypes 
    # can be updated. 
    # Otherwise, error.


    # # Before
    # # print(type(MyData))       # Print the type of MyData
    # # print(MyData)             # print the data frame
    # # print(MyData.head(20))
    # # print("\n")
    # # print(MyData.head())
    # # print("\n")
    # # print(MyData.describe())  # summary statistics on numeric columns
    # # print("\n")
    # # print(MyData.isnull().sum())   # how many nulls per column
    # # print("\n")
    # # print(MyData.info())
    # # print("\n")
    # print(MyData.dtypes)    # Print the data types of each column in MyData
    # # print("\n")
    # # # print all columns
    # # print(MyData.columns)   # print all columns
    # # print("\n")


    # # update FamSize datatype from float to integer
    # MyData['FamSize'] = MyData['FamSize'].astype(int)

    # # update Parch datatype from float to integer
    # MyData['Parch'] = MyData['Parch'].astype(int)

    # # update Survived datatype from float to integer
    # MyData['Survived'] = MyData['Survived'].astype(int)

    # # After
    # # print(type(MyData))       # Print the type of MyData
    # # print(MyData)             # print the data frame
    # # print(MyData.head(20))
    # # print("\n")
    # # print(MyData.head())
    # # print("\n")
    # # print(MyData.describe())  # summary statistics on numeric columns
    # # print("\n")
    # # print(MyData.isnull().sum())   # how many nulls per column
    # # print("\n")
    # # print(MyData.info())
    # # print("\n")
    # print(MyData.dtypes)    # Print the data types of each column in MyData
    # # print("\n")
    # # # print all columns
    # # print(MyData.columns)   # print all columns
    # # print("\n")







    # --
    # save the cleaned dataframe to a new csv file
    MyData.to_csv('MyNew_CleanFile_mod_7_proj_w_name.csv', index=False)
    print("Data saved to MyNew_CleanFile_mod_7_proj_w_name.csv")













def Project_6_clean_wits_trade_summary_dataset(args):
    # =======
    # Module 6
    # CS332 Winter 2025
    # 2/17/25
    # Tony Chan


    import seaborn as sns  # seaborn
    # Apply the default theme
    sns.set_theme()

    import pandas as pd
    import matplotlib.pyplot as plt
    # Show all columns
    pd.set_option('display.max_columns', 120)



    # =======
    # dataset 3 wits

    # # This is the path on local machine to data file
    # dirty
    # filename="C:/Users/Tony/OneDrive/Documents/CS332_Intro_to_Applied_Data_Science/Week_1/Module_1-Individual_Project_Assignment/wits_en_trade_summary_allcountries_allyears/en_USA_AllYears_WITS_Trade_Summary.csv"

    # clean
    # filename="C:/Users/Tony/OneDrive/Documents/CS332_Intro_to_Applied_Data_Science/Week_1/Module_1-Individual_Project_Assignment/wits_en_trade_summary_allcountries_allyears/MyNew_CleanFile.csv"
    # fix simple average

    # filename="C:/Users/Tony/OneDrive/Documents/CS332_Intro_to_Applied_Data_Science/Week_1/Module_1-Individual_Project_Assignment/wits_en_trade_summary_allcountries_allyears/MyNew_CleanFile_fix_sim_ave.csv"
    # print("filename: ", filename)
    # print("\n")

    # # --
    # # (a) Write Python code to assure that your datasets are in record format so that 
    # # they are structured as rows and columns, where each column has a variable name.
    # MyData = pd.read_csv(filename)
    # print("Data loaded successfully")
    # print(type(MyData))     # Print the type of MyData

    # # print the data frame
    # print(MyData)
    # print(MyData.head(15))

    # print(MyData.head())   # for summary statistics
    # print(MyData.describe())   # for summary statistics
    # print(MyData.info())   # for summary statistics

    # print(MyData.isnull().sum())   # for summary statistics)
    # print(MyData.dtypes)      # Print the data types of each column in MyData
    # print(MyData.columns)     # for summary statistics)
    # print("\n")




    # # --
    # # (c) Write Python code to find, count, report, and then clean any missing values.
    # # need to do (c) first then (b)
    # # Before
    # print(MyData.isnull().sum())   # for summary statistics)
    # print("\n")
    # print(MyData.info())   # for summary statistics
    # print("\n")
    # print(MyData.dtypes)    # Print the data types of each column in MyData
    # print("\n")
    # # print all columns
    # print(MyData.columns)   # print all columns


    # --
    # change column "Export" values that have commas to no commas
    # MyData['Export'] = MyData['Export'].str.replace(',', '')


    # # change column "Export" null values to the mean for that country. Look at 
    # # column "Country" and make mean for Export column for this country
    # MyData['Export'] = MyData['Export'].fillna(MyData.groupby('Country')['Export'].transform('mean'))

    # extract row with column "Reporter" = "United States" and column "Partner" = "World" and column "Product categories" = "All Products"
    # and "Indicator Type" = "Export"
    # MyData_US = MyData[(MyData['Reporter'] == 'United States') & (MyData['Partner'] == 'World') & (MyData['Product categories'] == 'All Products') & (MyData['Indicator Type'] == 'Export')]
    # print(MyData_US)


    # # delete columns 1990, 1989, 1988
    # MyData = MyData.drop(['1990', '1989', '1988'], axis=1)


    # # when value is null for column = 2021 to 1988 then calculates the mean across the columns for each row. 
    # # Do not use columns Reporter, Partner, Product categories, Indicator Type, Indicator
    # MyData['2021'] = MyData['2021'].fillna(MyData.iloc[:, 5:33].mean(axis=1))
    # MyData['2020'] = MyData['2020'].fillna(MyData.iloc[:, 5:33].mean(axis=1))
    # MyData['2019'] = MyData['2019'].fillna(MyData.iloc[:, 5:33].mean(axis=1))
    # MyData['2018'] = MyData['2018'].fillna(MyData.iloc[:, 5:33].mean(axis=1))
    # MyData['2017'] = MyData['2017'].fillna(MyData.iloc[:, 5:33].mean(axis=1))
    # MyData['2016'] = MyData['2016'].fillna(MyData.iloc[:, 5:33].mean(axis=1))


    # # there are still missing values in the dataset
    # # drop rows with missing values
    # MyData = MyData.dropna(subset=['2021', '2020', '2019', '2018', '2017', '2016'])



    # # changing datatypes need to be done after cleaning missing values
    # # change column "2021" datatype from float to int
    # MyData['2021'] = MyData['2021'].astype(int)






    # # --
    # # save the cleaned dataframe to a new csv file
    # MyData.to_csv('MyNew_CleanFile.csv', index=False)
    # print("Data saved to MyNew_CleanFile.csv")


    # # --
    # # After
    # print(MyData.isnull().sum())   # how many nulls
    # print("\n")
    # print(MyData.info())
    # print("\n")
    # print(MyData.dtypes)    # Print the data types of each column in MyData
    # print("\n")
    # # print all columns
    # print(MyData.columns)   # print all columns






    # # --
    # # (b) Write Python code to check and print the data types of the variables in your dataset. 
    # # Before
    # print(MyData.isnull().sum())   # how many nulls
    # print("\n")
    # print(MyData.info())
    # print("\n")
    # print(MyData.dtypes)    # Print the data types of each column in MyData
    # print("\n")

    # # print all columns
    # print(MyData.columns)   # print all columns

    # # Write code to correct any data types. For example, if Python reads in a categorical 
    # # variable as a number, you will need to update this to a category.
    # # change column "Export Product Share (%)" datatype from float to int
    # MyData['Export Product Share (%)'] = MyData['Export Product Share (%)'].astype(int)

    # # change column "AHS Simple Average (%)" datatype from float to int and round decimal to integer
    # MyData['AHS Simple Average (%)'] = MyData['AHS Simple Average (%)'].round().astype(int)

    # After
    # print(MyData.isnull().sum())   # how many nulls
    # print("\n")
    # print(MyData.info())
    # print("\n")
    # print(MyData.dtypes)    # Print the data types of each column in MyData
    # print("\n")




    # --
    # use clean file for the rest of the code. See top of file for filename
    # (d) Write Python code to find, report, and correct any incorrect values. You can use visual 
    # methods here. For example, you can "report" incorrect values visually.



    # --
    # dataset 3 wits
    # how to filter by "Reporter" = "United States" and "Partner" = "World" and "Product categories" = "All Products"
    # and "Indicator Type" = "Export" and "Indicator" = "Exports (in US$ Mil)"
    # from "United State" rows, plot column "Year" versus column "2021"
    # filter by "Reporter" = "United States" and "Partner" = "World" and "Product categories" = "All Products"
    # and "Indicator Type" = "Export" and "Indicator" = "Exports (in US$ Mil)"


    # # --
    # # MyData_US is for this plot. Filter out row.
    # MyData_US = MyData[(MyData['Reporter'] == 'United States') & (MyData['Partner'] == 'World') & (MyData['Product categories'] == 'All Products') & (MyData['Indicator Type'] == 'Export') & (MyData['Indicator'] == 'Exports (in US$ Mil)')]
    # print(MyData_US)

    # # plot column 1991 through 2021 versus row "Exports (in US$ Mil)"
    # # how to reverse order of x axis
    # plt.plot(MyData_US.columns[5:], MyData_US.iloc[0, 5:])
    # plt.xlabel('Year')
    # plt.ylabel('Exports (in US$ Mil)')
    # plt.title('Year vs Exports (in US$ Mil) for United States')
    # # Reverse the order on the x-axis
    # plt.gca().invert_xaxis()
    # # Display the plot
    # plt.show()

    # # --
    # # MyData_US is for this plot. Filter out row.
    # MyData_US = MyData[(MyData['Reporter'] == 'United States') & (MyData['Partner'] == 'World') & (MyData['Product categories'] == 'All Products') & (MyData['Indicator Type'] == 'Import') & (MyData['Indicator'] == 'Imports (in US$ Mil)')]
    # print(MyData_US)

    # # plot column 1991 through 2021 versus row "Imports (in US$ Mil)"
    # # how to reverse order of x axis
    # plt.plot(MyData_US.columns[5:], MyData_US.iloc[0, 5:])
    # plt.xlabel('Year')
    # plt.ylabel('Imports (in US$ Mil)')
    # plt.title('Year vs Imports (in US$ Mil) for United States')
    # # Reverse the order on the x-axis
    # plt.gca().invert_xaxis()
    # # Display the plot
    # plt.show()



    # # --
    # # MyData_US is for this plot. Filter out row.
    # MyData_US = MyData[(MyData['Reporter'] == 'United States') & (MyData['Partner'] == 'World') & (MyData['Product categories'] == 'All Products') & (MyData['Indicator Type'] == 'Tariff') & (MyData['Indicator'] == 'Simple Average (%)')]
    # print(MyData_US)

    # # plot column 1991 through 2021 versus row "Simple Average (%)"
    # # how to reverse order of x axis
    # plt.plot(MyData_US.columns[5:], MyData_US.iloc[0, 5:])
    # plt.xlabel('Year')
    # plt.ylabel('Simple Average (%)')
    # plt.title('Year vs Simple Average (%) for United States')
    # # Reverse the order on the x-axis
    # plt.gca().invert_xaxis()
    # # Display the plot
    # plt.show()




    # # --
    # # histogram plots

    # # --
    # # MyData_US is for this plot. Filter out row.
    # MyData_US = MyData[(MyData['Reporter'] == 'United States') & (MyData['Partner'] == 'World') & (MyData['Product categories'] == 'All Products') & (MyData['Indicator Type'] == 'Export') & (MyData['Indicator'] == 'Exports (in US$ Mil)')]
    # print(MyData_US)

    # # --
    # # plot histogram of column 1991 through 2021 versus row "Exports (in US$ Mil)"
    # plt.hist(MyData_US.iloc[0, 5:])
    # plt.xlabel('Exports (in US$ Mil)')
    # plt.ylabel('Frequency')
    # plt.title('Histogram of Exports (in US$ Mil) for United States')

    # # Display the plot
    # plt.show()


    # # --
    # # MyData_US is for this plot. Filter out row.
    # MyData_US = MyData[(MyData['Reporter'] == 'United States') & (MyData['Partner'] == 'World') & (MyData['Product categories'] == 'All Products') & (MyData['Indicator Type'] == 'Import') & (MyData['Indicator'] == 'Imports (in US$ Mil)')]
    # print(MyData_US)

    # # --
    # # plot histogram of column 1991 through 2021 versus row "Imports (in US$ Mil)"
    # plt.hist(MyData_US.iloc[0, 5:])
    # plt.xlabel('Imports (in US$ Mil)')
    # plt.ylabel('Frequency')
    # plt.title('Histogram of Imports (in US$ Mil) for United States')

    # # Display the plot
    # plt.show()


    # # --
    # # MyData_US is for this plot. Filter out row.
    # MyData_US = MyData[(MyData['Reporter'] == 'United States') & (MyData['Partner'] == 'World') & (MyData['Product categories'] == 'All Products') & (MyData['Indicator Type'] == 'Tariff') & (MyData['Indicator'] == 'Simple Average (%)')]
    # print(MyData_US)


    # # --
    # # plot histogram of column 1991 through 2021 versus row "Simple Average (%)"
    # plt.hist(MyData_US.iloc[0, 5:])
    # plt.xlabel('Simple Average (%)')
    # plt.ylabel('Frequency')
    # plt.title('Histogram of Simple Average (%) for United States')

    # # Display the plot
    # plt.show()




    # --
    # (e) Write Python code to find, report, and correct any values in the dataset(s) 
    # that might be correct but in the wrong format. For example, suppose you have a variable 
    # called "State" and the values can be state abbreviations like FL, OR, CA, etc. 
    # However, one of the entries is Fla. We know that this is FL and it needs to be updated 
    # to be the right (expected) format as prescribed by the dataset. Again, there are an 
    # infinite number of possibilities because all datasets are different. Take your time, 
    # explore your data, and determine how best to clean it.






    # # --
    # # MyData_US is for this plot. Filter out row.
    # MyData_US = MyData[(MyData['Reporter'] == 'United States') & (MyData['Partner'] == 'World') & (MyData['Product categories'] == 'All Products') & (MyData['Indicator Type'] == 'Export') & (MyData['Indicator'] == 'Exports (in US$ Mil)')]
    # # print(MyData_US)

    # # Before
    # # print(MyData.head(20))   # for summary statistics
    # # print("\n")
    # print(MyData_US.iloc[0, 5:])
    # print("\n")


    # # change row format to 1 decimal place from column 1991 to 2021
    # # MyData_US.iloc[0, 5:] = MyData_US.iloc[0, 5:].round(1)
    # MyData_US.iloc[0, 5:] = pd.to_numeric(MyData_US.iloc[0, 5:], errors='coerce').round(1)

    # # # After
    # # print(MyData.head(20))   # for summary statistics
    # # print("\n")
    # print(MyData_US.iloc[0, 5:])
    # print("\n")







    # # --
    # # MyData_US is for this plot. Filter out row.
    # MyData_US = MyData[(MyData['Reporter'] == 'United States') & (MyData['Partner'] == 'World') & (MyData['Product categories'] == 'All Products') & (MyData['Indicator Type'] == 'Import') & (MyData['Indicator'] == 'Imports (in US$ Mil)')]
    # print(MyData_US)


    # # Before
    # # print(MyData.head(20))   # for summary statistics
    # # print("\n")
    # print(MyData_US.iloc[0, 5:])
    # print("\n")


    # # change row format to 1 decimal place from column 1991 to 2021
    # # MyData_US.iloc[0, 5:] = MyData_US.iloc[0, 5:].round(1)
    # MyData_US.iloc[0, 5:] = pd.to_numeric(MyData_US.iloc[0, 5:], errors='coerce').round(1)

    # # # After
    # # print(MyData.head(20))   # for summary statistics
    # # print("\n")
    # print(MyData_US.iloc[0, 5:])
    # print("\n")








    # # --
    # # MyData_US is for this plot. Filter out row.
    # MyData_US = MyData[(MyData['Reporter'] == 'United States') & (MyData['Partner'] == 'World') & (MyData['Product categories'] == 'All Products') & (MyData['Indicator Type'] == 'Tariff') & (MyData['Indicator'] == 'Simple Average (%)')]
    # print(MyData_US)


    # # Before
    # # print(MyData.head(20))   # for summary statistics
    # # print("\n")
    # print(MyData_US.iloc[0, 5:])
    # print("\n")


    # # change row format to 1 decimal place from column 1991 to 2021
    # # MyData_US.iloc[0, 5:] = MyData_US.iloc[0, 5:].round(1)
    # MyData_US.iloc[0, 5:] = pd.to_numeric(MyData_US.iloc[0, 5:], errors='coerce').round(1)

    # # # After
    # # print(MyData.head(20))   # for summary statistics
    # # print("\n")
    # print(MyData_US.iloc[0, 5:])
    # print("\n")





    # --
    # (f) Write Python code to find, visualize, and correct any outliers.
    # Visualize outliers using box plots


    # # --
    # # MyData_US is for this plot. Filter out row.
    # MyData_US = MyData[(MyData['Reporter'] == 'United States') & (MyData['Partner'] == 'World') & (MyData['Product categories'] == 'All Products') & (MyData['Indicator Type'] == 'Export') & (MyData['Indicator'] == 'Exports (in US$ Mil)')]
    # print(MyData_US)


    # # box plot of row 'Exports (in US$ Mil)" from column 1991 to 2021
    # # log scale
    # plt.figure(figsize=(10, 5))
    # plt.boxplot(MyData_US.iloc[0, 5:])
    # plt.yscale('log')  # Set the y-axis to log scale
    # plt.xlabel('Exports (in US$ Mil)')
    # plt.title('Box plot of Exports (in US$ Mil) for United States')
    # plt.show()





    # # --
    # # MyData_US is for this plot. Filter out row.
    # MyData_US = MyData[(MyData['Reporter'] == 'United States') & (MyData['Partner'] == 'World') & (MyData['Product categories'] == 'All Products') & (MyData['Indicator Type'] == 'Import') & (MyData['Indicator'] == 'Imports (in US$ Mil)')]
    # print(MyData_US)


    # # # box plot of row "Imports (in US$ Mil)" from column 1991 to 2021
    # # # log scale
    # plt.figure(figsize=(10, 5))
    # plt.boxplot(MyData_US.iloc[0, 5:])
    # plt.yscale('log')  # Set the y-axis to log scale
    # plt.xlabel('Imports (in US$ Mil)')
    # plt.title('Box plot of Imports (in US$ Mil) for United States')
    # plt.show()





    # # --
    # # MyData_US is for this plot. Filter out row.
    # MyData_US = MyData[(MyData['Reporter'] == 'United States') & (MyData['Partner'] == 'World') & (MyData['Product categories'] == 'All Products') & (MyData['Indicator Type'] == 'Tariff') & (MyData['Indicator'] == 'Simple Average (%)')]
    # print(MyData_US)


    # # # box plot of row "Simple Average (%)" from column 1991 to 2021
    # # # linear scale
    # plt.figure(figsize=(10, 5))
    # plt.boxplot(MyData_US.iloc[0, 5:])
    # plt.xlabel('Simple Average (%)')
    # plt.title('Box plot of Simple Average (%) for United States')
    # plt.show()




    # # --
    # # MyData_US is for this plot. Filter out row.
    # MyData_US_export = MyData[(MyData['Reporter'] == 'United States') & (MyData['Partner'] == 'World') & (MyData['Product categories'] == 'All Products') & (MyData['Indicator Type'] == 'Export') & (MyData['Indicator'] == 'Exports (in US$ Mil)')]
    # # print(MyData_US_export)

    # # --
    # # MyData_US is for this plot. Filter out row.
    # MyData_US_import = MyData[(MyData['Reporter'] == 'United States') & (MyData['Partner'] == 'World') & (MyData['Product categories'] == 'All Products') & (MyData['Indicator Type'] == 'Import') & (MyData['Indicator'] == 'Imports (in US$ Mil)')]
    # # print(MyData_US_import)


    # # # scatter plot between "Export" and "Import" with log scale
    # # # log scale
    # plt.scatter(MyData_US_export.iloc[0, 5:], MyData_US_import.iloc[0, 5:])
    # plt.xlabel('Exports (in US$ Mil)')
    # plt.ylabel('Imports (in US$ Mil)')
    # plt.title('Exports vs Imports for United States')
    # plt.xscale('log')
    # plt.yscale('log')
    # plt.show()


    # # --
    # # remove outliers
    # # remove rows where "Exports" is greater than 70000
    # MyData = MyData[MyData['Export'] <= 70000]

    # # remove rows where "Imports" is greater than 70000
    # MyData = MyData[MyData['Import'] <= 70000]


    # # scatter plot between "Export" and "Import" with log scale
    # # log scale
    # plt.scatter(MyData['Export'], MyData['Import'])
    # plt.xlabel('Export')
    # plt.ylabel('Import')
    # plt.title('Export vs Import')
    # plt.xscale('log')
    # plt.yscale('log')
    # plt.show()





    # # --
    # # (g) Write Python code to create a new dataframe from one of your datasets such that 
    # # the new dataframe is normalized using min-max. Be sure to include a screen image 
    # # of the normalized dataset in your report.

    # from sklearn.preprocessing import MinMaxScaler

    # # --
    # # MyData_US is for this plot. Filter out row.
    # MyData_US_export = MyData[(MyData['Reporter'] == 'United States') & (MyData['Partner'] == 'World') & (MyData['Product categories'] == 'All Products') & (MyData['Indicator Type'] == 'Export') & (MyData['Indicator'] == 'Exports (in US$ Mil)')]
    # # print(MyData_US_export)

    # # --
    # # MyData_US is for this plot. Filter out row.
    # MyData_US_import = MyData[(MyData['Reporter'] == 'United States') & (MyData['Partner'] == 'World') & (MyData['Product categories'] == 'All Products') & (MyData['Indicator Type'] == 'Import') & (MyData['Indicator'] == 'Imports (in US$ Mil)')]
    # # print(MyData_US_import)



    # # # Before
    # # print(MyData.isnull().sum())   # how many nulls
    # # print("\n")
    # # print(MyData.info())
    # # print("\n")
    # # print(MyData.dtypes)    # Print the data types of each column in MyData
    # # print("\n")
    # print(MyData_US_import.head(15))
    # print("\n")
    # print(MyData_US_export.head(15))
    # print("\n")


    # # Initialize the MinMaxScaler
    # scaler = MinMaxScaler()

    # # Normalize the data for export row from columns 1991 to 2021
    # export_values = MyData_US_export.iloc[0, 5:].values.reshape(-1, 1)
    # normalized_export = scaler.fit_transform(export_values).flatten()

    # # Normalize the data for import row from columns 1991 to 2021
    # import_values = MyData_US_import.iloc[0, 5:].values.reshape(-1, 1)
    # normalized_import = scaler.fit_transform(import_values).flatten()

    # # Update the dataframe with normalized values
    # MyData_US_export.iloc[0, 5:] = normalized_export
    # MyData_US_import.iloc[0, 5:] = normalized_import

    # # Print the normalized data
    # print("Normalized Export Data:")
    # print(MyData_US_export.iloc[0, 5:])

    # print("\nNormalized Import Data:")
    # print(MyData_US_import.iloc[0, 5:])
    # print("\n")
    # print("new dataset normalized_Export and normalized_Import created")
    # print("\n")


    # # --

    # # scatter plot between "Export (US$ Thousand)" and "Import (US$ Thousand)" with log scale
    # # log scale
    # plt.scatter(normalized_df['Export (US$ Thousand)'], normalized_df['Import (US$ Thousand)'])
    # plt.xlabel('Export (US$ Thousand)')
    # plt.ylabel('Import (US$ Thousand)')
    # plt.title('Export (US$ Thousand) vs Import (US$ Thousand)')

    # plt.yscale('log')   # Set the y-axis to log scale
    # plt.xscale('log')   # Set the x-axis to log scale
    # # Display the plot
    # plt.show()


    # # scatter plot between "Export (US$ Thousand)" and "Import (US$ Thousand)" with linear scale
    # # linear scale
    # plt.scatter(normalized_df['Export (US$ Thousand)'], normalized_df['Import (US$ Thousand)'])
    # plt.xlabel('Export (US$ Thousand)')
    # plt.ylabel('Import (US$ Thousand)')
    # plt.title('Export (US$ Thousand) vs Import (US$ Thousand)')

    # # plt.yscale('log')   # Set the y-axis to log scale
    # # plt.xscale('log')   # Set the x-axis to log scale
    # # Display the plot
    # plt.show()









    # # --
    # # (h) Write Python code to create a new dataframe that contains only unlabeled and 
    # # quantitative data. 

    # # "country" column is qualitative. so, "country" column will not be included in new dataset. 
    # # New dataset just contains numerical columns.

    # # Before
    # print(MyData.isnull().sum())   # how many nulls
    # print("\n")
    # print(MyData.info())
    # print("\n")
    # print(MyData.dtypes)    # Print the data types of each column in MyData
    # print("\n")
    # print(MyData.head(15))
    # print("\n")

    # # Select only the columns with numeric data types
    # MyData_quant = MyData.select_dtypes(include=['int64', 'float64'])

    # # Display the new dataset
    # # After
    # print(MyData_quant.isnull().sum())   # how many nulls
    # print("\n")
    # print(MyData_quant.info())
    # print("\n")
    # print(MyData_quant.dtypes)    # Print the data types of each column in MyData
    # print("\n")
    # print(MyData_quant.head(15))
    # print("\n")
    # print("new dataset MyData_quant created")
    # print("\n")

















def Project_6_clean_kaggle_world_export_import_dataset(args):
    # =======
    # Module 6
    # CS332 Winter 2025
    # 2/17/25
    # Tony Chan
    # =======
    # dataset 2 Kaggle World Export Import
    # This is the path on local machine to data file
    filename="C:/Users/Tony/OneDrive/Documents/CS332_Intro_to_Applied_Data_Science/Week_1/Module_1-Individual_Project_Assignment/kaggle_world_import_export/34_years_world_export_import_dataset.csv"

    # dataset 2 Kaggle India Exports
    # This is the path on local machine to data file
    filename="C:/Users/Tony/OneDrive/Documents/CS332_Intro_to_Applied_Data_Science/Week_1/Module_1-Individual_Project_Assignment/kaggle_Exports and Imports of India(1997-July 2022)/exports and imports of india(1997- July 2022) - exports and imports.csv"
    filename="C:/Users/Tony/OneDrive/Documents/CS332_Intro_to_Applied_Data_Science/Week_1/Module_1-Individual_Project_Assignment/kaggle_Exports and Imports of India(1997-July 2022)/MyNew_CleanFile.csv"
    print("filename: ", filename)
    print("\n")
    
    # # --
    # # (a) Write Python code to assure that your datasets are in record format so that 
    # # they are structured as rows and columns, where each column has a variable name.
    # MyData = pd.read_csv(filename)
    # print("Data loaded successfully")

    # # print the data frame
    # print(MyData)
    # print(MyData.head(15))
    # # print(type(MyData))     # Print the type of MyData

    # print(MyData.head())   # for summary statistics
    # print(MyData.describe())   # for summary statistics
    # print(MyData.info())   # for summary statistics

    # print(MyData.isnull().sum())   # for summary statistics)
    # print(MyData.dtypes)   # for summary statistics
    # print(MyData.columns)   # for summary statistics)
    # print("\n")


    # # --
    # # (c) Write Python code to find, count, report, and then clean any missing values.
    # # need to do (c) first then (b)
    # # Before
    # print(MyData.isnull().sum())   # for summary statistics)
    # print("\n")
    # print(MyData.info())   # for summary statistics
    # print("\n")

    # --
    # change column "Export" values that have commas to no commas
    # MyData['Export'] = MyData['Export'].str.replace(',', '')

    # # change column "Export" to float
    # MyData['Export'] = MyData['Export'].astype(float)

    # # change column "Export" null values to the mean for that country. Look at 
    # # column "Country" and make mean for Export column for this country
    # MyData['Export'] = MyData['Export'].fillna(MyData.groupby('Country')['Export'].transform('mean'))

    # --
    # change column "Import" values that have commas to no commas
    # MyData['Import'] = MyData['Import'].str.replace(',', '')

    # # change column "Import" to float
    # MyData['Import'] = MyData['Import'].astype(float)

    # # change column "Import" null values to the mean for that country. Look at
    # # column "Country" and make mean for Import column for this country
    # MyData['Import'] = MyData['Import'].fillna(MyData.groupby('Country')['Import'].transform('mean'))

    # --
    # change column "Total Trade" values that have commas to no commas
    # MyData['Total Trade'] = MyData['Total Trade'].str.replace(',', '')

    # # change column "Total Trade" to float
    # MyData['Total Trade'] = MyData['Total Trade'].astype(float)

    # # change column "Total Trade" null values to the mean for that country. Look at
    # # column "Country" and make mean for Total Trade column for this country
    # MyData['Total Trade'] = MyData['Total Trade'].fillna(MyData.groupby('Country')['Total Trade'].transform('mean'))

    # --
    # change column "Trade Balance" values that have commas to no commas
    # MyData['Trade Balance'] = MyData['Trade Balance'].str.replace(',', '')

    # # change column "Trade Balance" to float
    # MyData['Trade Balance'] = MyData['Trade Balance'].astype(float)

    # # change column "Trade Balance" null values to the mean for that country. Look at
    # # column "Country" and make mean for Trade Balance column for this country
    # MyData['Trade Balance'] = MyData['Trade Balance'].fillna(MyData.groupby('Country')['Trade Balance'].transform('mean'))

    # # --
    # # if column "Total Trade" = 0 or null then delete row
    # MyData = MyData[MyData['Total Trade'] != 0]
    # MyData = MyData.dropna(subset=['Total Trade'])

    # # --
    # # save the cleaned dataframe to a new csv file
    # MyData.to_csv('MyNew_CleanFile.csv', index=False)
    # print("Data saved to MyNew_CleanFile.csv")


    # # --
    # # After
    # print(MyData.isnull().sum())   # for summary statistics)
    # print("\n")
    # print(MyData.info())   # for summary statistics
    # print("\n")



    # # --
    # # (b) Write Python code to check and print the data types of the variables in your dataset. 
    # # Before
    # print(type(MyData))     # Print the type of MyData
    # print("\n")
    # print(MyData.dtypes)    # Print the data types of each column in MyData
    # print("\n")

    # # Write code to correct any data types. For example, if Python reads in a categorical 
    # # variable as a number, you will need to update this to a category.
    # # change column "Export Product Share (%)" datatype from float to int
    # MyData['Export Product Share (%)'] = MyData['Export Product Share (%)'].astype(int)

    # # change column "AHS Simple Average (%)" datatype from float to int and round decimal to integer
    # MyData['AHS Simple Average (%)'] = MyData['AHS Simple Average (%)'].round().astype(int)

    # After
    # # print(MyData.dtypes)    # Print the data types of each column in MyData
    # # print("\n")



    # --
    # use clean file for the rest of the code. See top of file for filename
    # (d) Write Python code to find, report, and correct any incorrect values. You can use visual 
    # methods here. For example, you can "report" incorrect values visually.

    # --
    # dataset 2 Kaggle India Exports
    # how to filter by "Country" = "U S A"
    # from "U S A" rows, plot column "Financial Year(start)" versus column "Import"
    # filter by "Country" = "U S A"
    # MyData_US = MyData[MyData['Country'] == 'U S A']
    # print(MyData_US)

    # # plot column "Year" versus column "Import (US$ Thousand)"
    # plt.plot(MyData_US['Financial Year(start)'], MyData_US['Import'])
    # plt.xlabel('Financial Year(start)')
    # plt.ylabel('Import')
    # plt.title('Financial Year vs Import for United States to India')

    # # Display the plot
    # plt.show()

    # # plot column "Year" versus column "Export (US$ Thousand)"
    # plt.plot(MyData_US['Financial Year(start)'], MyData_US['Export'])
    # plt.xlabel('Financial Year(start)')
    # plt.ylabel('Export')
    # plt.title('Financial Year vs Export for United States to India')

    # # Display the plot
    # plt.show()

    # --
    # # plot for "Country" = "U S A" for column "Financial Year(start)" versus column "Trade Balance"
    # # filter by "Country" = "U S A"
    # MyData_US = MyData[MyData['Country'] == 'U S A']
    # print(MyData_US)

    # # plot column "Year" versus column "Trade Balance"
    # plt.plot(MyData_US['Financial Year(start)'], MyData_US['Trade Balance'])
    # plt.xlabel('Financial Year(start)')
    # plt.ylabel('Trade Balance')
    # plt.title('Financial Year vs Trade Balance for United States to India')

    # # Display the plot
    # plt.show()

    # # --
    # # plot histogram of column "Export (US$ Thousand)")
    # plt.hist(MyData_US['Export'])
    # plt.xlabel('Export')
    # plt.ylabel('Frequency')
    # plt.title('Histogram of Export for United States to India')

    # # Display the plot
    # plt.show()

    # # plot histogram of column "Import (US$ Thousand)")
    # plt.hist(MyData_US['Import'])
    # plt.xlabel('Import')
    # plt.ylabel('Frequency')
    # plt.title('Histogram of Import for United States to India')

    # # Display the plot
    # plt.show()


    # # --
    # # plot histogram for "Country" = "U S A" for column "Financial Year(start)" versus column "Trade Balance"
    # # filter by "Country" = "U S A"
    # MyData_US = MyData[MyData['Country'] == 'U S A']
    # print(MyData_US)

    # # plot histogram of column "Trade Balance"
    # plt.hist(MyData_US['Trade Balance'])
    # plt.xlabel('Trade Balance')
    # plt.ylabel('Frequency')
    # plt.title('Histogram of Trade Balance for United States to India')

    # # Display the plot
    # plt.show()






    # # --
    # # (e) Write Python code to find, report, and correct any values in the dataset(s) 
    # # that might be correct but in the wrong format. For example, suppose you have a variable 
    # # called "State" and the values can be state abbreviations like FL, OR, CA, etc. 
    # # However, one of the entries is Fla. We know that this is FL and it needs to be updated 
    # # to be the right (expected) format as prescribed by the dataset. Again, there are an 
    # # infinite number of possibilities because all datasets are different. Take your time, 
    # # explore your data, and determine how best to clean it.

    # # Before
    # print(MyData.head(20))   # for summary statistics
    # print("\n")

    # # change column "Export" format to 1 decimal place
    # MyData['Export'] = MyData['Export'].round(1)

    # # change column "Import" format to 1 decimal place
    # MyData['Import'] = MyData['Import'].round(1)

    # # change column "Total Trade" format to 1 decimal place
    # MyData['Total Trade'] = MyData['Total Trade'].round(1)

    # # change column "Trade Balance" format to 1 decimal place
    # MyData['Trade Balance'] = MyData['Trade Balance'].round(1)

    # # After
    # print(MyData.head(20))   # for summary statistics
    # print("\n")





    # # --
    # # (f) Write Python code to find, visualize, and correct any outliers.
    # # Visualize outliers using box plots

    # # box plot of column "Export"
    # # log scale
    # plt.figure(figsize=(10, 5))
    # plt.boxplot(MyData['Export'])
    # plt.yscale('log')  # Set the y-axis to log scale
    # plt.xlabel('Export')
    # plt.title('Box plot of Export')
    # plt.show()

    # # box plot of column "Import"
    # # log scale
    # plt.figure(figsize=(10, 5))
    # plt.boxplot(MyData['Import'])
    # plt.yscale('log')  # Set the y-axis to log scale
    # plt.xlabel('Import')
    # plt.title('Box plot of Import')
    # plt.show()


    # # box plot of column "Total Trade"
    # # log scale
    # plt.figure(figsize=(10, 5))
    # plt.boxplot(MyData['Total Trade'])
    # plt.yscale('log')  # Set the y-axis to log scale
    # plt.xlabel('Total Trade')
    # plt.title('Box plot of Total Trade')
    # plt.show()

    # # scatter plot between "Export" and "Import" with log scale
    # # log scale
    # plt.scatter(MyData['Export'], MyData['Import'])
    # plt.xlabel('Export')
    # plt.ylabel('Import')
    # plt.title('Export vs Import')
    # # plt.xscale('log')
    # # plt.yscale('log')
    # plt.show()




    # # --
    # # remove outliers
    # # remove rows where "Exports" is greater than 70000
    # MyData = MyData[MyData['Export'] <= 70000]

    # # remove rows where "Imports" is greater than 70000
    # MyData = MyData[MyData['Import'] <= 70000]


    # # scatter plot between "Export" and "Import" with log scale
    # # log scale
    # plt.scatter(MyData['Export'], MyData['Import'])
    # plt.xlabel('Export')
    # plt.ylabel('Import')
    # plt.title('Export vs Import')
    # plt.xscale('log')
    # plt.yscale('log')
    # plt.show()



    # # --
    # # (g) Write Python code to create a new dataframe from one of your datasets such that 
    # # the new dataframe is normalized using min-max. Be sure to include a screen image 
    # # of the normalized dataset in your report.

    # from sklearn.preprocessing import MinMaxScaler

    # # before
    # print(MyData[['Export', 'Import']].head(15))
    # print("\n")

    # # Initialize the MinMaxScaler
    # scaler = MinMaxScaler()

    # # Normalize the data for columns 'Export' and 'Import'
    # # normalized_data = scaler.fit_transform(MyData)    # all columns. Must be numeric
    # normalized_data = scaler.fit_transform(MyData[['Export', 'Import']])
    # print(normalized_data)
    # print("\n")

    # # Create a new DataFrame with the normalized data
    # normalized_df = pd.DataFrame(normalized_data, columns=['Export', 'Import'])

    # # Display the normalized DataFrame
    # # after
    # print(normalized_df.head(15))
    # print("\n")

    # # scatter plot between "Export (US$ Thousand)" and "Import (US$ Thousand)" with log scale
    # # log scale
    # plt.scatter(normalized_df['Export (US$ Thousand)'], normalized_df['Import (US$ Thousand)'])
    # plt.xlabel('Export (US$ Thousand)')
    # plt.ylabel('Import (US$ Thousand)')
    # plt.title('Export (US$ Thousand) vs Import (US$ Thousand)')

    # plt.yscale('log')   # Set the y-axis to log scale
    # plt.xscale('log')   # Set the x-axis to log scale
    # # Display the plot
    # plt.show()


    # # scatter plot between "Export (US$ Thousand)" and "Import (US$ Thousand)" with linear scale
    # # linear scale
    # plt.scatter(normalized_df['Export (US$ Thousand)'], normalized_df['Import (US$ Thousand)'])
    # plt.xlabel('Export (US$ Thousand)')
    # plt.ylabel('Import (US$ Thousand)')
    # plt.title('Export (US$ Thousand) vs Import (US$ Thousand)')

    # # plt.yscale('log')   # Set the y-axis to log scale
    # # plt.xscale('log')   # Set the x-axis to log scale
    # # Display the plot
    # plt.show()





    # # --
    # # (h) Write Python code to create a new dataframe that contains only unlabeled and 
    # # quantitative data. 

    # # "country" column is qualitative. so, "country" column will not be included in new dataset. 
    # # New dataset just contains numerical columns.

    # # Before
    # print(MyData.head(15))

    # # Select only the columns with numeric data types
    # MyData_quant = MyData.select_dtypes(include=['int64', 'float64'])

    # # Display the new dataset
    # # After
    # print(MyData_quant.head(15))




def Project_6_clean_kaggle_india_trade_dataset(args):
    # =======
    # dataset 1 Kaggle World Export Import
    # # dataset 1 Kaggle World Export Import
    # # This is the path on local machine to data file
    # filename="C:/Users/Tony/OneDrive/Documents/CS332_Intro_to_Applied_Data_Science/Week_1/Module_1-Individual_Project_Assignment/kaggle_world_import_export/34_years_world_export_import_dataset.csv"

    # # dataset 2 Kaggle India Exports
    # # This is the path on local machine to data file
    # filename="C:/Users/Tony/OneDrive/Documents/CS332_Intro_to_Applied_Data_Science/Week_1/Module_1-Individual_Project_Assignment/kaggle_Exports and Imports of India(1997-July 2022)/exports and imports of india(1997- July 2022) - exports and imports.csv"
    # filename="C:/Users/Tony/OneDrive/Documents/CS332_Intro_to_Applied_Data_Science/Week_1/Module_1-Individual_Project_Assignment/kaggle_Exports and Imports of India(1997-July 2022)/MyNew_CleanFile.csv"
    print("filename: ", args.filename)
    print("\n")
    
    # --
    # # how to filter by "Partner Name" = "United States"
    # # from "United State" rows, plot column "Year" versus column "Import (US$ Thousand)"
    # # filter by "Partner Name" = "United States"
    # MyData_US = MyData[MyData['Partner Name'] == 'United States']
    # print(MyData_US)

    # # plot column "Year" versus column "Import (US$ Thousand)"
    # plt.plot(MyData_US['Year'], MyData_US['Import (US$ Thousand)'])
    # plt.xlabel('Year')
    # plt.ylabel('Import (US$ Thousand)')
    # plt.title('Year vs Import (US$ Thousand) for United States')

    # # Display the plot
    # plt.show()

    # # plot column "Year" versus column "Export (US$ Thousand)"
    # plt.plot(MyData_US['Year'], MyData_US['Export (US$ Thousand)'])
    # plt.xlabel('Year')
    # plt.ylabel('Export (US$ Thousand)')
    # plt.title('Year vs Export (US$ Thousand) for United States')

    # # Display the plot
    # plt.show()


    # # plot histogram of column "Export (US$ Thousand)")
    # plt.hist(MyData_US['Export (US$ Thousand)'])
    # plt.xlabel('Export (US$ Thousand)')
    # plt.ylabel('Frequency')
    # plt.title('Histogram of Export (US$ Thousand) for United States')

    # # Display the plot
    # plt.show()


    # # plot histogram of column "Import (US$ Thousand)")
    # plt.hist(MyData_US['Import (US$ Thousand)'])
    # plt.xlabel('Import (US$ Thousand)')
    # plt.ylabel('Frequency')
    # plt.title('Histogram of Import (US$ Thousand) for United States')

    # # Display the plot
    # plt.show()


    # # # plot column "Year" versus column "AHS Simple Average (%)"
    # plt.plot(MyData_US['Year'], MyData_US['AHS Simple Average (%)'])
    # plt.xlabel('Year')
    # plt.ylabel('AHS Simple Average (%)')
    # plt.title('Year vs AHS Simple Average (%) for United States')

    # # Display the plot
    # plt.show()


    # # plot histogram of column "AHS Simple Average (%)")
    # plt.hist(MyData_US['AHS Simple Average (%)'])
    # plt.xlabel('AHS Simple Average (%)')
    # plt.ylabel('Frequency')
    # plt.title('Histogram of AHS Simple Average (%) for United States')

    # # Display the plot
    # plt.show()


    # --
    # (e) Write Python code to find, report, and correct any values in the dataset(s) 
    # that might be correct but in the wrong format. For example, suppose you have a variable 
    # called "State" and the values can be state abbreviations like FL, OR, CA, etc. 
    # However, one of the entries is Fla. We know that this is FL and it needs to be updated 
    # to be the right (expected) format as prescribed by the dataset. Again, there are an 
    # infinite number of possibilities because all datasets are different. Take your time, 
    # explore your data, and determine how best to clean it.

    # print(MyData.head(20))   # for summary statistics
    # print("\n")

    # # change column "AHS Simple Average (%)" format to 1 decimal place
    # MyData['AHS Simple Average (%)'] = MyData['AHS Simple Average (%)'].round(1)

    # # change column "AHS MaxRate (%)" format to 1 decimal place
    # MyData['AHS MaxRate (%)'] = MyData['AHS MaxRate (%)'].round(1)

    # # change column "AHS MinRate (%)" format to 1 decimal place
    # MyData['AHS MinRate (%)'] = MyData['AHS MinRate (%)'].round(1)

    # print(MyData.head(20))   # for summary statistics
    # print("\n")


    # --
    # (f) Write Python code to find, visualize, and correct any outliers.
    # Visualize outliers using box plots

    # # box plot of column "Export (US$ Thousand)"
    # # log scale
    # plt.figure(figsize=(10, 5))
    # plt.boxplot(MyData['Export (US$ Thousand)'])
    # plt.yscale('log')  # Set the y-axis to log scale
    # plt.xlabel('Export (US$ Thousand)')
    # plt.title('Box plot of Export (US$ Thousand)')
    # plt.show()

    # # box plot of column "Import (US$ Thousand)"
    # # log scale
    # plt.figure(figsize=(10, 5))
    # plt.boxplot(MyData['Import (US$ Thousand)'])
    # plt.yscale('log')  # Set the y-axis to log scale
    # plt.xlabel('Import (US$ Thousand)')
    # plt.title('Box plot of Import (US$ Thousand)')
    # plt.show()


    # # scatter plot between "Export (US$ Thousand)" and "Import (US$ Thousand)" with log scale
    # # log scale
    # plt.scatter(MyData['Export (US$ Thousand)'], MyData['Import (US$ Thousand)'])
    # plt.xlabel('Export (US$ Thousand)')
    # plt.ylabel('Import (US$ Thousand)')
    # plt.title('Export (US$ Thousand) vs Import (US$ Thousand)')
    # plt.xscale('log')
    # plt.yscale('log')
    # plt.show()


    # # scatter plot between "Export (US$ Thousand)" and "Import (US$ Thousand)" with log scale
    # # linear scale
    # plt.scatter(MyData['Export (US$ Thousand)'], MyData['Import (US$ Thousand)'])
    # plt.xlabel('Export (US$ Thousand)')
    # plt.ylabel('Import (US$ Thousand)')
    # plt.title('Export (US$ Thousand) vs Import (US$ Thousand)')
    # # plt.xscale('log')
    # # plt.yscale('log')
    # plt.show()

    # # --
    # # MFN MaxRate (%) and MFN MinRate (%) for all countries and years
    # # MFN MaxRate (%) versus years for all countries
    # plt.scatter(MyData['Year'], MyData['MFN MaxRate (%)'])
    # plt.xlabel('Year')
    # plt.ylabel('MFN MaxRate (%)')
    # plt.title('MFN MaxRate (%) versus Year')
    # plt.show()

    # # MFN MinRate (%) versus years for all countries
    # plt.scatter(MyData['Year'], MyData['MFN MinRate (%)'])
    # plt.xlabel('Year')
    # plt.ylabel('MFN MinRate (%)')
    # plt.title('MFN MinRate (%) versus Year')
    # plt.show()

    # # box plot of column "MFN MaxRate (%)"
    # # log scale
    # plt.figure(figsize=(10, 5))
    # plt.boxplot(MyData['MFN MaxRate (%)'])
    # # plt.yscale('log')  # Set the y-axis to log scale
    # plt.xlabel('MFN MaxRate (%)')
    # plt.title('Box plot of MFN MaxRate (%)')
    # plt.show()

    # # histogram of column "MFN MaxRate (%)"
    # plt.hist(MyData['MFN MaxRate (%)'])
    # plt.xlabel('MFN MaxRate (%)')
    # plt.ylabel('Frequency')
    # plt.title('Histogram of MFN MaxRate (%)')
    # plt.show()


    # # --
    # # remove outliers
    # # remove rows where "MFN MaxRate (%)" is greater than 3001
    # MyData = MyData[MyData['MFN MaxRate (%)'] <= 3001]

    # # # remove rows where "MFN MinRate (%)" is greater than 3001
    # # MyData = MyData[MyData['MFN MinRate (%)'] <= 3001]

    # # remove rows where "MFN MaxRate (%)" is less than 0
    # MyData = MyData[MyData['MFN MaxRate (%)'] >= 0]

    # # # remove rows where "MFN MinRate (%)" is less than 0
    # # MyData = MyData[MyData['MFN MinRate (%)'] >= 0]

    # # histogram of column "MFN MaxRate (%)"
    # plt.hist(MyData['MFN MaxRate (%)'])
    # plt.xlabel('MFN MaxRate (%)')
    # plt.ylabel('Frequency')
    # plt.title('Histogram of MFN MaxRate (%)')
    # plt.show()

    # # save the cleaned dataframe to a new csv file
    # MyData.to_csv('MyNew_CleanFile.csv', index=False)
    # print("Data saved to MyNew_CleanFile.csv")







    # # --
    # # (g) Write Python code to create a new dataframe from one of your datasets such that 
    # # the new dataframe is normalized using min-max. Be sure to include a screen image 
    # # of the normalized dataset in your report.

    # from sklearn.preprocessing import MinMaxScaler

    # # print columns "Export (US$ Thousand)" and "Import (US$ Thousand)"
    # # before
    # print(MyData[['Export (US$ Thousand)', 'Import (US$ Thousand)']].head(15))
    # print("\n")

    # # Initialize the MinMaxScaler
    # scaler = MinMaxScaler()

    # # Normalize the data for columns 'Export (US$ Thousand)' and 'Import (US$ Thousand)'
    # # normalized_data = scaler.fit_transform(MyData)    # all columns. Must be numeric
    # normalized_data = scaler.fit_transform(MyData[['Export (US$ Thousand)', 'Import (US$ Thousand)']])
    # print(normalized_data)
    # print("\n")

    # # Create a new DataFrame with the normalized data
    # normalized_df = pd.DataFrame(normalized_data, columns=['Export (US$ Thousand)', 'Import (US$ Thousand)'])

    # # Display the normalized DataFrame
    # # after
    # print(normalized_df.head(15))
    # print("\n")

    # # scatter plot between "Export (US$ Thousand)" and "Import (US$ Thousand)" with log scale
    # # log scale
    # plt.scatter(normalized_df['Export (US$ Thousand)'], normalized_df['Import (US$ Thousand)'])
    # plt.xlabel('Export (US$ Thousand)')
    # plt.ylabel('Import (US$ Thousand)')
    # plt.title('Export (US$ Thousand) vs Import (US$ Thousand)')

    # plt.yscale('log')   # Set the y-axis to log scale
    # plt.xscale('log')   # Set the x-axis to log scale
    # # Display the plot
    # plt.show()


    # # scatter plot between "Export (US$ Thousand)" and "Import (US$ Thousand)" with linear scale
    # # linear scale
    # plt.scatter(normalized_df['Export (US$ Thousand)'], normalized_df['Import (US$ Thousand)'])
    # plt.xlabel('Export (US$ Thousand)')
    # plt.ylabel('Import (US$ Thousand)')
    # plt.title('Export (US$ Thousand) vs Import (US$ Thousand)')

    # # plt.yscale('log')   # Set the y-axis to log scale
    # # plt.xscale('log')   # Set the x-axis to log scale
    # # Display the plot
    # plt.show()


    # # --
    # # (h) Write Python code to create a new dataframe that contains only unlabeled and 
    # # quantitative data. 

    # # "country" column is qualitative. so, "country" column will not be included in new dataset. 
    # # New dataset just contains numerical columns.

    # # Before
    # print(MyData.head(15))

    # # Select only the columns with numeric data types
    # MyData_quant = MyData.select_dtypes(include=['int64', 'float64'])

    # # Display the new dataset
    # # After
    # print(MyData_quant.head(15))


    # --
    # # (i) Finally, take any further steps you feel are needed to clean up your data such as 
    # # discretization and feature generation.

def Module_6_demo_class_balance_with_smote():
    # =======

    # from Module 6 explora balance example
    # determine if data is imbalance
    # use frequency count to determine if data is imbalance
    # if data is imbalance, use SMOTE to balance the data
    import pandas as pd
    from sklearn.preprocessing import MinMaxScaler

    ## !! You will need to update this to be YOUR path to your filename
    filename="C:/Users/Tony/OneDrive/Documents/CS332_Intro_to_Applied_Data_Science/Week_1/Module_1-Individual_Project_Assignment/kaggle_world_import_export/34_years_world_export_import_dataset.csv"
    print("filename: ", filename)
    print("\n")
    MyDF=pd.read_csv(filename)
    print(MyDF)


    # # oversampling
    # from imblearn.over_sampling import SMOTE
    # from collections import Counter

    # # Separate the features and the target variable
    # X = MyDF.drop('GPA', axis=1)
    # y = MyDF['GPA']

    # # Apply the SMOTE algorithm
    # smote = SMOTE(random_state=0)
    # X_resampled, y_resampled = smote.fit_resample(X, y)

    # # Print the class distribution of the resampled data
    # print(Counter(y_resampled))

    # # # undersampling
    # from imblearn.under_sampling import RandomUnderSampler

    # # Apply the RandomUnderSampler
    # rus = RandomUnderSampler(random_state=0)
    # X_resampled, y_resampled = rus.fit_resample(X, y)

    # # Print the class distribution of the resampled data
    # print(Counter(y_resampled))

    # # # combined sampling
    # from imblearn.combine import SMOTEENN

    # # Apply the SMOTEENN
    # smote_enn = SMOTEENN(random_state=0)
    # X_resampled, y_resampled = smote_enn.fit_resample(X, y)

    # # Print the class distribution of the resampled data
    # print(Counter(y_resampled))



    # =======

    # from Module 6 explora normalization example
    # from medium exercise 1

    # import pandas as pd

    # from medium article

    # remove NaN values
    # df = pd.read_csv('datasets/dpc-covid19-ita-regioni.csv')
    # df.dropna(axis=1, inplace=True)
    # df.head(7)

    # # Single Feature Scaling
    # df['tamponi'] = df['tamponi'] / df['tamponi'].max()
    # print(df)

    # # Min-Max
    # df['totale_casi'] = (df['totale_casi'] - df['totale_casi'].min()) / (df['totale_casi'].max() - df['totale_casi'].min())
    # print(df)

    # # Z-Score
    # df['deceduti'] = (df['deceduti'] - df['deceduti'].mean()) / df['deceduti'].std()
    # print(df)

    # # Log Scaling
    # import numpy as np
    # df['dimessi_guariti'] = df['dimessi_guariti'].apply(lambda x: np.log(x) if x != 0 else 0)
    # print(df)

    # # Clipping
    # vmax = 10000
    # vmin = 10
    # df['ricoverati_con_sintomi'] = df['ricoverati_con_sintomi'].apply(
    #     lambda x: vmax if x > vmax else vmin if x < vmin else x)
    # print(df)




def Module_6_demo_normalization_and_discretization():
    # # from Module 6 explora normal example
    # # from sklearn.preprocessing import MinMaxScaler
    # import pandas as pd
    # from sklearn.preprocessing import MinMaxScaler

    # ## !! You will need to update this to be YOUR path to your filename
    # filename="C:/Users/Tony/OneDrive/Documents/CS332_Intro_to_Applied_Data_Science/Week_6/explora_normal_example/Mod6_Small_Labeled_Quant_SummerStudents.csv"
    # print("filename: ", filename)
    print("\n")
    
    # MyDF=pd.read_csv(filename)
    # print(MyDF)
    # ## Use the MinMax Normallization method
    # scaler = MinMaxScaler() 
    # MyDF[['GPA', 'TestScore', 'WritingScore']] = scaler.fit_transform(MyDF[['GPA', 'TestScore', 'WritingScore']]) 

    # print(MyDF)


    # # from Module 6 explora normal example
    # # discretization
    # import pandas as pd
    # filename="C:/Users/profa/Desktop/C_O/OSU/DS Course Dev/Datasets/Mod6_Small_Labeled_Quant_SummerStudents.csv"
    # DF=pd.read_csv(filename)
    # print(DF)
    # ## Discretization
    # group_names = ['Low',  'Medium', 'High', 'Very High']
    # # Specify our bin boundaries
    # evaluation_bins = [0, 2.8, 3.5, 3.7, 4 ]
    # # And stitch it all together
    # DF['GPA'] = pd.cut(DF['GPA'], bins = evaluation_bins, labels = group_names, include_lowest = True)
    # print(DF)



    # # from Module 6 explora normal example
    # # feature engineering
    # # new feature creation testscore + writingscore
    # import pandas as pd
    # filename="C:/Users/Tony/OneDrive/Documents/CS332_Intro_to_Applied_Data_Science/Week_6/explora_normal_example/Mod6_Small_Labeled_Quant_SummerStudents.csv"
    # DF=pd.read_csv(filename)
    # print(DF)
    # DF["TestSum"]=DF["TestScore"]+DF["WritingScore"]
    # print(DF)





def Module_5_titanic_clean_and_cluster():
    # =======

    # Module 5 Discussion 5
    # CS332 Winter 2025
    # 2/5/25
    # Tony Chan

    # ======

    # Module 2 Exercise 2
    # CS332 Winter 2025
    # 1/19/25
    # Tony Chan


    # import seaborn as sns  # seaborn
    # # Apply the default theme
    # sns.set_theme()

    # import pandas as pd
    # import matplotlib.pyplot as plt
    # # Show all columns
    # pd.set_option('display.max_columns', 120)

    # # kaggle 02 dataset
    # # This is the path on local machine to data file
    # filename="C:/Users/Tony/OneDrive/Documents/CS332_Intro_to_Applied_Data_Science/Week_1/Module_1-Individual_Project_Assignment/kaggle_world_import_export/34_years_world_export_import_dataset.csv"
    # print("filename: ", filename)
    print("\n")

    # # From Module 2 Exercise

    # # 1. load the data file
    # MyData = pd.read_csv(filename)
    # print("Data loaded successfully")

    # # 2. print the data frame
    # print(MyData)
    # print(MyData.head(15))
    # print(type(MyData))

    # print(MyData.head())   # for summary statistics
    # print(MyData.describe())   # for summary statistics
    # print(MyData.info())   # for summary statistics

    # print(MyData.isnull().sum())   # for summary statistics)
    # print(MyData.dtypes)   # for summary statistics
    # print(MyData.columns)   # for summary statistics)


    # # delete first 3 columns
    # MyData.drop(MyData.columns[0:3], axis=1, inplace=True)
    # print("First 3 columns deleted")
    # print(MyData.head(15))

    # # delete column name "Sex"
    # MyData.drop(columns=['Sex'], inplace=True)
    # print("Column 'Sex' deleted")
    # print(MyData.head(15))

    # # delete next one columns
    # MyData.drop(MyData.columns[1:2], axis=1, inplace=True)
    # print(MyData.head(15))

    # # delete next two columns
    # MyData.drop(MyData.columns[2:5], axis=1, inplace=True)
    # print(MyData.head(15))

    # # start two columns to the right
    # MyData = MyData.iloc[:, 2:]
    # print(MyData.head(15))

    # # delete next two columns
    # MyData.drop(MyData.columns[3:5], axis=1, inplace=True)
    # print(MyData.head(15))

    # print("head")
    # MyData.head()   # for summary statistics
    # print("\n")
    # print("describe")
    # MyData.describe()   # for summary statistics
    # print("\n")
    # print("info")
    # MyData.info()   # for summary statistics
    # print("\n")
    # print("isnull")
    # MyData.isnull().sum()   # for summary statistics
    # print("\n")
    # print("dtypes")
    # # print all the data types
    # print(MyData.dtypes)
    # print("\n")


    # # save the cleaned dataframe to a new csv file
    # MyData.to_csv('MyNew_Titanic_3columns.csv', index=False)

    # --
    # # chart 3 variables Name, Age, Fare
    # # Create a scatter plot of the Age and Fare columns,
    # # with the points colored by the cluster labels
    # plt.scatter(MyData['Age'], MyData['Fare'])
    # plt.xlabel('Age')
    # plt.ylabel('Fare')
    # plt.title('Age vs Fare')
    # plt.show()

    # # bar chart Name vesus Age
    # plt.bar(MyData['Name'], MyData['Age'])
    # plt.xlabel('Name')
    # plt.ylabel('Age')
    # plt.title('Name vs Age')
    # plt.show()

    # # line chart Name vesus Fare
    # # sort lowest fare to highest fare
    # MyData.sort_values(by=['Fare'], inplace=True)
    # plt.plot(MyData['Name'], MyData['Fare'])
    # plt.xlabel('Name')
    # plt.ylabel('Fare')
    # plt.title('Name vs Fare')
    # plt.show()

    # --
    # clean up the data

    # # delete duplicate rows
    # MyData.drop_duplicates(inplace=True)
    # print("Duplicate rows deleted")
    # print(MyData.head(15))

    # # replace missing values in the Age column with the mean of the column
    # MyData.loc[:, 'Age'] = MyData['Age'].fillna(MyData['Age'].mean())
    # print(MyData.head(15))

    # # 11. Check the age is between 1 thru 120 and replace with the mean if not
    # MyData['Age'] = MyData['Age'].apply(lambda x: MyData['Age'].mean() 
    #                                     if x < 1 or x > 120 else x)
    # print(MyData.head(25))

    # # replace missing values in the Fare column with the median of the column
    # MyData.loc[:, 'Fare'] = MyData['Fare'].fillna(MyData['Fare'].median())
    # print(MyData.head(15))

    # # 13. Check the Fare is between 0 thru 8000 and replace with the median if not
    # MyData['Fare'] = MyData['Fare'].apply(lambda x: MyData['Fare'].median() 
    #                                       if x < 0 or x > 8000 else x)
    # print(MyData.head(25))

    # # replace 0 values in the Fare column with the median of the column
    # MyData.loc[:, 'Fare'] = MyData['Fare'].apply(lambda x: MyData['Fare'].median() if x == 0 else x)
    # print(MyData.head(15))

    # # replace 0 or negative values in the age column with the mean of the column
    # MyData.loc[:, 'Age'] = MyData['Age'].apply(lambda x: MyData['Age'].mean() if x <= 0 else x)
    # print(MyData.head(15))

    # # replace 0 or negative values in the Fare column with the median of the column
    # MyData.loc[:, 'Fare'] = MyData['Fare'].apply(lambda x: MyData['Fare'].median() if x <= 0 else x)
    # print(MyData.head(15))

    # # Check for any non-numeric values or NaNs in the 'Age' column
    # print(MyData['Age'].isnull().sum())  # Should be 0 if there are no NaNs

    # # Convert the 'Age' column to integers
    # MyData['Age'] = MyData['Age'].round().astype(int)
    # print(MyData.head(15))

    # # Check for any non-numeric values or NaNs in the 'Fare' column
    # print(MyData['Fare'].isnull().sum())  # Should be 0 if there are no NaNs


    # # round the Fare column to 2 decimal places
    # MyData['Fare'] = MyData['Fare'].round(2)
    # print(MyData.head(15))

    # print("head")
    # print(MyData.head(15))   # for summary statistics
    # print("\n")
    # print("describe")
    # MyData.describe()   # for summary statistics
    # print("\n")
    # print("info")
    # MyData.info()   # for summary statistics
    # print("\n")
    # print("isnull")
    # MyData.isnull().sum()   # for summary statistics
    # print("\n")
    # print("dtypes")
    # # print all the data types
    # print(MyData.dtypes)
    # print("\n")



    # # Remove titles from names
    # MyData['Name'] = MyData['Name'].str.replace('Mr.', '')
    # MyData['Name'] = MyData['Name'].str.replace('Mrs.', '')
    # MyData['Name'] = MyData['Name'].str.replace('Miss.', '')

    # # Replace 'Mr.', 'Mrs.', and 'Miss.' with an empty string
    # MyData['Name'] = MyData['Name'].str.replace('Mr.', '').str.replace('Mrs.', '').str.replace('Miss.', '')

    # Remove text within parentheses


    # Remove extra spaces and commas


    # # Split the cleaned names into first and last names

    # print("head")
    # print(MyData.head(15))   # for summary statistics
    # print("\n")
    # print("describe")
    # MyData.describe()   # for summary statistics
    # print("\n")
    # print("info")
    # MyData.info()   # for summary statistics
    # print("\n")
    # print("isnull")
    # MyData.isnull().sum()   # for summary statistics
    # print("\n")
    # print("dtypes")
    # # print all the data types
    # print(MyData.dtypes)
    # print("\n")

    # # save the cleaned dataframe to a new csv file
    # MyData.to_csv('MyNew_Titanic_CleanFile.csv', index=False)
    # print("Data saved to MyNew_Titanic_CleanFile.csv")

    # =======
    # After cleaning do charts again

    # --
    # # chart 3 variables Name, Age, Fare
    # # Create a scatter plot of the Age and Fare columns,
    # # with the points colored by the cluster labels
    # plt.scatter(MyData['Age'], MyData['Fare'])
    # plt.xlabel('Age')
    # plt.ylabel('Fare')
    # plt.title('Age vs Fare')
    # plt.show()

    # # bar chart Name vesus Age
    # plt.bar(MyData['Name'], MyData['Age'])
    # plt.xlabel('Name')
    # plt.ylabel('Age')
    # plt.title('Name vs Age')
    # plt.show()

    # # line chart Name versus Fare
    # # sort lowest fare to highest fare
    # MyData.sort_values(by=['Fare'], inplace=True)
    # plt.plot(MyData['Name'], MyData['Fare'])
    # plt.xlabel('Name')
    # plt.ylabel('Fare')
    # plt.title('Name vs Fare')
    # plt.show()

    # ======

    # # convert country variable data type to string
    # MyData['Country'] = MyData['Country'].astype(str)

    # # print all the data types
    # print(MyData.dtypes)
    # # line feed
    # print("\n")

    # Access the first cell under the 'Country' column
    # first_country_value = MyData['Country'].iloc[0]

    # # Print the value and its datatype
    # print(first_country_value)
    # print(type(first_country_value))

    # # remove all non numeric columns
    # # loop through the columns
    # for col in MyData.columns:
    #     if (MyData[col].dtype == 'object'):
    #         # if data type is object, remove the column
    #         MyData.drop(columns=[col], inplace=True)

    # # print(MyData)
    # # print all the data types
    # print(MyData.dtypes)
    # # line feed
    # print("\n")


    # # remove first 20 columns
    # MyData.drop(MyData.columns[0:20], axis=1, inplace=True)        
    # print("3 quantatative columns remaining")
    # # print all the data types
    # print(MyData.dtypes)
    # # line feed
    # print("\n")
    # print(MyData.head(15))


    # # save the cleaned dataframe to a new csv file
    # MyData.to_csv('MyNew_Cancer_CleanFile.csv', index=False)


    # # Convert the quantitative variable to a qualitative variable
    # MyData['Health_Infrastructure_Index'] = MyData['Health_Infrastructure_Index'].apply(lambda x: 'Low' if x <= 6 else 'High')

    # print("2 quantatative and 1 qualitative columns remaining")
    # # print all the data types
    # print(MyData.dtypes)
    # # line feed
    # print("\n")
    # print(MyData.head(15))

    # # Change the name of the 'Health_Infrastructure_Index' column to "LABEL"
    # MyData.rename(columns={'Health_Infrastructure_Index': 'LABEL'}, inplace=True)

    # print("2 quantatative and 1 qualitative LABEL columns remaining")
    # # print all the data types
    # print(MyData.dtypes)
    # # line feed
    # print("\n")
    # print(MyData.head(15))


    # # save the cleaned dataframe to a new csv file
    # MyData.to_csv('MyNew_Cancer_LABEL_CleanFile.csv', index=False)




    # Module 2 Exercise

    # # 1. Removing all non-numeric columns.
    # # loop through the columns
    # for col in MyData.columns:
    #     if (MyData[col].dtype == 'object'):
    #         # check if the column is not numeric
    #         # check rows that are numeric and count < 5 then delete column
    #         if (MyData[col].str.isnumeric().sum() < 5):
    #             # delete column
    #             MyData.drop(columns=[col], inplace=True)
    #             print(MyData)
        
    # # 2. Aggregate SibSp and Parch by summing them and 
    # # putting the sum into a new column called Sib_Parch
    # MyData['Sib_Parch'] = MyData['SibSp'] + MyData['Parch']
    # print(MyData.head(15))



    # # 2a. remove SibSp and Parch.
    # MyData.drop(columns=['SibSp', 'Parch'], inplace=True)
    # print(MyData.head(15))

    # # 3. Remove Pclass
    # MyData.drop(columns=['Pclass'], inplace=True)
    # print(MyData.head(15))

    # # 4. Take the "Survived" column off the dataframe and save it as a LIST.
    # Survived = MyData['Survived'].tolist()
    # print(Survived)
    # print(MyData.head(15))

    # # 5. Remove the "Survived" column from the dataframe.
    # MyData.drop(columns=['Survived'], inplace=True)
    # print(MyData.head(15))

    # # # 6. Add the "Survived" column back to the dataframe.
    # # MyData['Survived'] = Survived
    # # print(MyData.head(15))

    # # 7. Remove "Ticket" column
    # MyData.drop(columns=['Ticket'], inplace=True)
    # print(MyData.head(15))

    # # 8. Remove "PassengerId" column
    # MyData.drop(columns=['PassengerId'], inplace=True)
    # print(MyData.head(25))

    # # # 8a. Remove "Survived" column
    # # # it needs more cleaning
    # # MyData.drop(columns=['Survived'], inplace=True)
    # # print(MyData.head(15))

    # # --
    # # 3 remaining columns: Age, Fare, Sib_Parch, Survived
    # # 9. assuring that there are no missing values in the dataframe
    # print(MyData.isnull().sum())

    # # 10. Replace missing values in the Age column with the mean of the column
    # MyData['Age'].fillna(MyData['Age'].mean(), inplace=True)
    # print(MyData.head(25))

    # # 11. Check the age is between 1 thru 120 and replace with the mean if not
    # MyData['Age'] = MyData['Age'].apply(lambda x: MyData['Age'].mean() 
    #                                     if x < 1 or x > 120 else x)
    # print(MyData.head(25))

    # # 12. Replace missing values in the Fare column with the median of the column
    # MyData['Fare'].fillna(MyData['Fare'].median(), inplace=True)
    # print(MyData.head(25))

    # # 13. Check the Fare is between 0 thru 8000 and replace with the median if not
    # MyData['Fare'] = MyData['Fare'].apply(lambda x: MyData['Fare'].median() 
    #                                       if x < 0 or x > 8000 else x)
    # print(MyData.head(25))



    # # 14. Replace missing values in the Sib_Parch column with the mode of the column
    # MyData['Sib_Parch'].fillna(MyData['Sib_Parch'].mode()[0], inplace=True)
    # print(MyData.head(25))

    # #14a. Check the Sib_Parch is greater than 0 and replace with the mode if not
    # MyData['Sib_Parch'] = MyData['Sib_Parch'].apply(lambda x: MyData['Sib_Parch'].mode()[0]
    #                                                 if x < 0 else x)
    # print(MyData.head(25))

    # # 9a. assuring that there are no missing values in the dataframe
    # print(MyData.isnull().sum())


    # # 15. Save the cleaned dataframe to a new csv file
    # MyData.to_csv('MyNew_Titanitc_CleanFile.csv', index=False)

    # # --
    # # do kmeans clustering from skit-learn
    # from sklearn.cluster import KMeans
    # from sklearn.preprocessing import StandardScaler


    # # 16. Standardize the data
    # scaler = StandardScaler()
    # X = scaler.fit_transform(MyData)

    # # 17. Create a KMeans model with 2 clusters
    # kmeans = KMeans(n_clusters=2, random_state=0)

    # # 18. Fit the model to the data
    # kmeans.fit(X)

    # # 19. Get the cluster centers
    # centers = kmeans.cluster_centers_
    # print(centers)

    # # 20. Get the cluster labels
    # labels = kmeans.labels_

    # # 21. Add the cluster labels to the dataframe
    # MyData['Cluster'] = labels
    # print(MyData.head(15))

    # # 22. Save the dataframe with the cluster labels to a new csv file
    # MyData.to_csv('MyNew_Titanitc_CleanFile_Clustered.csv', index=False)

    # # 23. Create a scatter plot of the Age and Fare columns,
    # # with the points colored by the cluster labels
    # plt.scatter(MyData['Age'], MyData['Fare'], c=MyData['Cluster'])
    # plt.xlabel('Age')
    # plt.ylabel('Fare')
    # plt.title('Age vs Fare')
    # plt.show()


    # # Assuming kmeans is your KMeans model
    # centroids = kmeans.cluster_centers_
    # print("Cluster centroids:\n", centroids)

    # # Add cluster labels to the DataFrame
    # MyData['Cluster'] = kmeans.labels_

    # # Group by cluster and calculate summary statistics
    # cluster_summary = MyData.groupby('Cluster').mean()
    # print("Cluster summary:\n", cluster_summary)



    # 24. Create a scatter plot of the Age and Sib_Parch columns,

    # # decisiontreeclassifier
    # from sklearn.tree import DecisionTreeClassifier
    # from sklearn.model_selection import train_test_split
    # from sklearn.metrics import accuracy_score

    # # 25. Split the data into training and testing sets
    # #X_train, X_test, y_train, y_test = train_test_split(X, MyData['Survived'], test_size=0.2, random_state=0)
    # # no Survived column
    # X_train, X_test, y_train, y_test = train_test_split(X, MyData['Cluster'], test_size=0.2, random_state=0)

    # # 26. Create a DecisionTreeClassifier model
    # dt = DecisionTreeClassifier(random_state=0)

    # # 27. Fit the model to the training data
    # dt.fit(X_train, y_train)

    # # 28. Predict the labels of the test data
    # y_pred = dt.predict(X_test)

    # # 29. Calculate the accuracy of the model
    # accuracy = accuracy_score(y_test, y_pred)
    # print('Accuracy:', accuracy)

    # # 30. Save the accuracy to a text file
    # with open('accuracy.txt', 'w') as f:
    #     f.write(str(accuracy))
    
    # # 31. Create a bar plot of the feature importances
    # importances = dt.feature_importances_
    # plt.bar(MyData.columns, importances)
    # plt.xlabel('Feature')
    # plt.ylabel('Importance')
    # plt.title('Feature Importances')
    # plt.show()

    # # 32. Save the plot to a new file

def Module_2_explore_student_program_data(args):
    # =======
    # From Module 2 Discussion

    # Module 2 Discussion 2
    # CS332 Winter 2025
    # Due 1/17/25
    # Tony Chan


    # # This is the path on local machine to data file
    # # C:\Users\Tony\OneDrive\Documents\CS332_Intro_to_Applied_Data_Science\Week_2\Module 2 - Discussion_ Python Practice and Peer Review\Module_2_explore_student_program_data
    # import seaborn as sns  # seaborn
    # # Apply the default theme
    # sns.set_theme()

    # import pandas as pd
    # import matplotlib.pyplot as plt
    # filename = "C:/Users/Tony/OneDrive/Documents/CS332_Intro_to_Applied_Data_Science/Week_2/Module 2 - Discussion_ Python Practice and Peer Review/Module_2_explore_student_program_data/StudentSummerProgramDataClean_DT.csv"
    # print("filename: ", filename")
    print("\n")

    # # 1. load the data file
    # MyData = pd.read_csv(filename)
    # print("Data file loaded")

    # # 2. print the data frame
    # print("Printing the data frame")
    # print(MyData)
    # # print(type(MyData))

    # MyData.head()   # for summary statistics

    # # 3. find number of columns
    # print("number of columns\n", len(MyData.columns))    

    # # 4. find number of rows
    # print("number of rows\n", len(MyData.index))

    # # 5. loop through the column names and print them
    # # print(MyData.columns.tolist())  # alternate print column names list without loop
    # # loop through the columns
    # for col1 in MyData.columns:
    #     print(col1)

    # # 6. find third column
    # print("Third column values in ", MyData.columns[2])
    # # loop through third column rows using panda commands
    # for i in MyData.iloc[:,2]:
    #     print(i)
    
    # # 7. mean average of gpas. this is the third column
    # # Extract the third column (index 2)
    # third_column = MyData.iloc[:, 2]      # panda

    # # Calculate the mean
    # mean = third_column.mean()

    # # Print the mean
    # print("Mean of the third column: ", mean)


    # # 8. summary statistics
    # # Extract the third column (index 2)
    # third_column = MyData.iloc[:, 2]

    # # Print summary statistics
    # print("Summary statistics of the third column:")
    # print(third_column.describe())    

    # # 9. Create a bar chart
    # # Check if the 'Decision' column exists
    # if 'Decision' in MyData.columns:
    #     # Create the catplot
    #     # visualizing categorical data.
    #     sns.catplot(data=MyData, x="Decision", kind="count")
    
    #     # Display the plot
    #     plt.show()
    # else:
    #     print("The column 'Decision' does not exist in the DataFrame.")


# =======
# call functions from here


def main():
    args = None  # Pass any arguments if needed
    # Project_7_clean_kaggle_world_trade_data_kaggle_world_trade_data(args)
    # Project_7_run_kmeans_trade_clustering(args)
    Project_9_run_decision_tree_export_model(args)


if __name__ == "__main__":
    main()
    

    


