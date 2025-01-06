import os

def get_test_folder():
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'tests')

def get_results_folder():
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'tests', 'test_results')
