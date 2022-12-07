from application import decision_tree_appservice as tree_appservice
from application import k_algorithms_appservice as k_appservice
from application import regression_appservice as reg_appservice
from application import repository_appservice as repo_appservice
from application import naives_bayes_appservice as NB_appservice

repo_appservice.clear_temp()

tdf = repo_appservice.get_treated_dataset()


linear_regression = reg_appservice.linear(tdf, df_test_size= 0.3)
linear_regression.print_overview()

logistic_regression = reg_appservice.logistic(tdf, df_test_size= 0.3)
logistic_regression.print_overview()

decision_tree = tree_appservice.decision_tree(tdf, tree_depth= 10, df_test_size= 0.3, open_file= True)
decision_tree.print_overview()

knn = k_appservice.knn(tdf, num_neighbors= 10, df_test_size= 0.3)
knn.print_overview()

kmeans = k_appservice.kmeans(tdf, num_clusters= None, dimensions=0)
kmeans.print_overview()

naives_bayes = NB_appservice.naives_bayes(tdf, df_test_size=0.3)
naives_bayes.print_overview()