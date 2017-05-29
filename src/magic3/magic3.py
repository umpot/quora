import pandas as pd
import networkx as nx





def run_kcore():
    df_train = pd.read_csv(DATA_DIR + "train.csv", usecols=["qid1", "qid2"])

    df_test = pd.read_csv(DATA_DIR + "test_with_ids.csv", usecols=["qid1", "qid2"])

    df_all = pd.concat([df_train, df_test])

    print("df_all.shape:", df_all.shape) # df_all.shape: (2750086, 2)

    df = df_all

    g = nx.Graph()

    g.add_nodes_from(df.qid1)

    edges = list(df[['qid1', 'qid2']].to_records(index=False))

    g.add_edges_from(edges)

    g.remove_edges_from(g.selfloop_edges())

    print(len(set(df.qid1)), g.number_of_nodes()) # 4789604

    print(len(df), g.number_of_edges()) # 2743365 (after self-edges)

    df_output = pd.DataFrame(data=g.nodes(), columns=["qid"])

    print("df_output.shape:", df_output.shape)

    NB_CORES = 20

    for k in range(2, NB_CORES + 1):

        fieldname = "kcore{}".format(k)

        print("fieldname = ", fieldname)

        ck = nx.k_core(g, k=k).nodes()

        print("len(ck) = ", len(ck))

        df_output[fieldname] = 0

        df_output.ix[df_output.qid.isin(ck), fieldname] = k

    df_output.to_csv("question_kcores.csv", index=None)
    def run_kcore_max():

    df_cores = pd.read_csv("question_kcores.csv", index_col="qid")

    df_cores.index.names = ["qid"]

    df_cores['max_kcore'] = df_cores.apply(lambda row: max(row), axis=1)

    df_cores[['max_kcore']].to_csv("question_max_kcores.csv") # with index