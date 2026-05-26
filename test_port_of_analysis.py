from inpythotools import load_mat_database
from novitrack import analyse_nttestrecord, results_nttestrecord

def test_analysis():
    testpath = r"C:/Users/alexa/OneDrive/GitHub\Heimel/InPythoTools/test_data/"
#    filename = testpath + r"nttestdb_examples_ou.mat"
    filename = testpath + r"nttestdb_examples.mat"
    db = load_mat_database(filename)
#    record = db.iloc[-1].to_dict()
    record = db.iloc[1]
    out = analyse_nttestrecord(record, verbose=False)
    results = results_nttestrecord(out)
    return results


if __name__ == "__main__":
    test_analysis()
