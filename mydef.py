from curses import raw
from sqlite3 import DateFromTicks
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

sns.set_theme()


def draw_missing_data_table(dataframe):
    total_missing_data = dataframe.isnull().sum()
    total_row_dataframe = dataframe.isnull().count()
    percent_missing_data = (total_missing_data / total_row_dataframe)
    missing_data_table = pd.concat([total_missing_data , percent_missing_data], axis=1, keys=['Total', 'Percent']).reset_index
    return missing_data_table

def fill_missing_data(df):
    from sklearn.impute import SimpleImputer
    impute = SimpleImputer(missing_values=np.nan, strategy='mean')
    
    list_missing_values_col=list(df.columns[df.isna().any()])
    for col in list_missing_values_col:
        # Get col index
        missing_values_col_index=df.columns.get_loc(col)
        filled_missing_values=impute.fit_transform(df.iloc[:, missing_values_col_index:missing_values_col_index+1 ].values)
        df[col] = pd.DataFrame(filled_missing_values)
    return df


def draw_bar_chart(x_axis,y_axis,x_name= None, y_name = None,x_axis_rotation = None):
    plt.bar(x = x_axis, height = y_axis, color = 'green')
    plt.xticks(x_axis,rotation = x_axis_rotation)
    plt.xlabel(x_name)
    plt.ylabel(y_name)
    plt.show()


def draw_pie(df, name =None):
    
    df = df.value_counts()
    #define Seaborn color palette to use
    colors = sns.color_palette('pastel')[0:10]
    #create pie chart
    plt.pie(
        x = df, 
        colors = colors, 
        labels=list(df.index),
        autopct='%1.2f%%',  
        radius= 1.5,
        )
    # x_axis_legend = -0,5
    # y_axis_legend = 0.2
    plt.legend(loc="lower right", bbox_to_anchor=(-0.5, 0.2))
    plt.title(
        name, 
        fontsize = 20 , 
        loc='left',
        x = -0.75)
    plt.show()


def draw_head_map(df, vmin =None, vmax = None):
    fig, ax = plt.subplots(figsize = (10,8))
    first_row =1
    last_column = -1
    numerical_df = df[(df.select_dtypes(include=['float64']).columns)]
    # ones_like can build a matrix of booleans with the same shape of our data
    numerical_corr = numerical_df.corr()
    # ones_like can build a matrix of booleans with the same shape of our data
    ones_corr = np.ones_like(numerical_corr, dtype=bool)
    # remvove first row and last column 
    numerical_corr = numerical_corr.iloc[first_row:, :last_column]
    # np.triu: retun only upper triangle matrix
    mask = np.triu(ones_corr)[first_row:, :last_column]

    sns.heatmap(
        data =numerical_corr ,
        mask = mask,
        # Show number 
        annot = True,
        # Round number
        fmt = ".2f",
        # Set color
        cmap ='winter_r',
        # Set limitation of color bar (right)
        vmin = vmin, vmax = vmax,
        # Color of the lines that will divide each cell.
        linecolor = 'white',
        # Width of the lines that will divide each cell.  
        linewidths = 0.5);
    yticks = [i.upper () for i in numerical_corr.index]
    xticks = [i.upper () for i in numerical_corr.columns]

    ax.set_yticklabels(yticks, rotation = 0, fontsize =8);
    ax.set_xticklabels(xticks, rotation = 0, fontsize =8);

    title = 'HEADMAP OF NUMERICAL VARIABLES'
    ax.set_title(title, loc ='left', fontsize = 20);


def draw_multiple_categorical_chart(df, hue = None):
    row_of_chart =1
    col_of_chart = 2
    list_categorical_column = list(df.select_dtypes(include=['object', 'int64']).columns)
    index = 0
    while index < len(list_categorical_column):
        fig, (ax1, ax2) = plt.subplots(row_of_chart, col_of_chart, figsize=(15,5))
        sns.countplot(data = df, x = list_categorical_column[index], hue = hue, palette='winter_r',ax =ax1)
        index +=1
        sns.countplot(data = df, x = list_categorical_column[index], hue = hue, palette='winter_r',ax =ax2)
        index +=1

def draw_multiple_numerical_chart(df, hue = None, type_of_chart = None):
    row_of_chart =1
    col_of_chart = 2
    list_numerical_column = list(df.select_dtypes(include=['float64']).columns)
    if type_of_chart =='his':
        index = 0
        while index < len(list_numerical_column ):
            fig, (ax1, ax2) = plt.subplots(row_of_chart, col_of_chart, figsize=(15,5))
            sns.histplot(data = df, x = list_numerical_column [index], hue = hue, kde=True, palette='winter',ax =ax1)
            index +=1
            sns.histplot(data = df, x = list_numerical_column [index], hue = hue, kde =True, palette='winter',ax =ax2)
            index +=1
    if type_of_chart=='kde':
        for col in list_numerical_column:
            plt.figure(figsize=(10,4));
            list_value_hue = list(df[hue].unique())
            sns.kdeplot(data = df[df[hue] ==int(list_value_hue[0])], x = col, shade= True, alpha = 1);
            sns.kdeplot(data = df[df[hue] ==int(list_value_hue[1])], x = col, shade= True );
            plt.legend (title = str(hue), labels= ['1','0']);


def score_module_classifier(x_train, y_train, x_test, y_test):
    # Import ML Libraries
    from sklearn.metrics import accuracy_score, recall_score, precision_score , confusion_matrix
    from xgboost import XGBClassifier
    from sklearn.linear_model import SGDClassifier
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.svm import SVC
    from lightgbm import LGBMClassifier
    from sklearn.naive_bayes import GaussianNB
    from sklearn.tree import DecisionTreeClassifier

    classifiers = [
        [XGBClassifier(random_state =1, use_label_encoder=False),'XGB Classifier'], [RandomForestClassifier(random_state =1),'Random Forest'], 
        [LGBMClassifier(random_state =1),'LGBM Classifier'], [KNeighborsClassifier(), 'K-Nearest Neighbours'], 
        [SGDClassifier(random_state =1),'SGD Classifier'], [SVC(random_state =1),'SVC'],
        [GaussianNB(),'GaussianNB'],[DecisionTreeClassifier(random_state =1),'Decision Tree Classifier']
    ];

    for cls in classifiers:
        model = cls[0]
        model.fit(x_train, y_train)
        
        y_pred = model.predict(x_test)

        accuracy =  round(accuracy_score(y_test, y_pred), 2) *  100
        recall = round(recall_score(y_test, y_pred), 2) *  100
        precision = round(precision_score(y_test, y_pred), 2) *  100

        print(f"{cls[1]}")
        print ('Confusion Matrix:')
        print(confusion_matrix(y_test, y_pred))
        print("Accuracy : ", accuracy)
        print("Recall : ", recall)
        print("Precision : ", precision)
        print("---------------------------------")


def SMOTE(x, y, xlabel_name =None):
    fig, (ax1, ax2) = plt.subplots(1, 2,figsize =(6,4))
    sns.barplot(x=['0', '1'], y =[sum(y == 0), sum(y == 1)], ax = ax1, palette='winter_r' );
    ax1.set_title("Before Oversampling")
    ax1.set_xlabel(xlabel_name)

    from imblearn.over_sampling import SMOTE

    sm = SMOTE(random_state = 2)
    x, y = sm.fit_resample(x, y)
    sns.barplot(x=['0', '1'], y =[sum(y == 0), sum(y == 1)], ax = ax2, palette='winter_r' );
    ax2.set_title("After Oversampling")
    ax2.set_xlabel(xlabel_name)

    plt.tight_layout()
    plt.show()
    return x,y


def draw_important_feature(features, x, y,x_train , y_train ):  
    # Import ML Libraries
    from sklearn.feature_selection import SelectFromModel
    from sklearn.metrics import accuracy_score, recall_score, precision_score , confusion_matrix
    from xgboost import XGBClassifier
    from sklearn.linear_model import SGDClassifier
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.svm import SVC
    from lightgbm import LGBMClassifier
    from sklearn.naive_bayes import GaussianNB
    from sklearn.tree import DecisionTreeClassifier

    classifiers = [
        [XGBClassifier(random_state =1, use_label_encoder=False, eval_metric='mlogloss'),'XGB Classifier'],
        [RandomForestClassifier(random_state =1),'Random Forest'], 
        [LGBMClassifier(random_state =1),'LGBM Classifier'], 
        [DecisionTreeClassifier(random_state =1),'Decision Tree Classifier']]

    for cls in classifiers:
        model = cls[0]
        model.fit(x_train, y_train)
        
        print(f"{cls}")
        best_feature = SelectFromModel(cls[0])
        best_feature.fit(x,y)

        transformedX = best_feature.transform(x)
        print(f"Old Shape: {x.shape} New shape: {transformedX.shape}")
        print("\n")

        imp_feature = pd.DataFrame({'Feature': features, 'Importance': model.feature_importances_})
        plt.figure(figsize=(10,4));
        plt.title("Feature Importance Graphic");
        plt.xlabel("importance ");
        plt.ylabel("features");
        plt.barh(list(imp_feature['Feature'].values), list(imp_feature['Importance'].values),     color = sns.color_palette());
        plt.show(); 