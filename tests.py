from tweets import *
import os
import unittest

class Tests(unittest.TestCase):
    
    def test_not_a_file(self):
        self.assertRaises(AssertionError, tweets_analysis, '')
        self.assertRaises(AssertionError, tweets_analysis, 999)
        self.assertRaises(AssertionError, tweets_analysis, 'aaa')
        
    def test_not_json_file(self):
        filename = 'temp_test_not_json.csv'
        with open(filename, 'w') as f:
            f.write('bla bla bla')
        self.assertRaises(Exception, tweets_analysis, filename)
        
    def test_json_contains_created_at_and_tag(self):
        filename = 'temp_test_wrong_json_format.json'
        with open(filename, 'w') as f:
            f.write("""[{
                "text": "bla",
                "retweeted": false,
                "tag": "#ai",
                "retweet_count": 0,
                "id": 1
            }]""")
        self.assertRaises(AssertionError, tweets_analysis, filename)
        
    def test_tag_is_list_of_strings(self):
        self.assertRaises(AssertionError, tweets_analysis, 'tweets.json', tags='#ai')    
    
    def test_unknown_tag(self):
        self.assertRaises(Exception, tweets_analysis, 'tweets.json', tags=['#robot'])
        
    def test_bin_frequency_format(self):
        self.assertRaises(AssertionError, tweets_analysis, 'tweets.json', bin_frequency='bla')
        self.assertRaises(AssertionError, tweets_analysis, 'tweets.json', bin_frequency='60m')
        self.assertRaises(AssertionError, tweets_analysis, 'tweets.json', bin_frequency='20.5s')
        self.assertRaises(AssertionError, tweets_analysis, 'tweets.json', bin_frequency='t20s')
        
    def test_stat_sign_format(self):
        self.assertRaises(AssertionError, tweets_analysis, 'tweets.json', stat_sign=-0.1)
        self.assertRaises(AssertionError, tweets_analysis, 'tweets.json', stat_sign=1.1)
        self.assertRaises(AssertionError, tweets_analysis, 'tweets.json', stat_sign='bla')
        
    def test_print_corr_format(self):
        self.assertRaises(AssertionError, tweets_analysis, 'tweets.json', print_corr='bla')
        self.assertRaises(AssertionError, tweets_analysis, 'tweets.json', print_corr=1)
        self.assertRaises(AssertionError, tweets_analysis, 'tweets.json', print_corr=0.1)
        
    def test_enough_data_correlation(self):
        self.assertRaises(AssertionError, tweets_analysis, 'tweets.json', tags=['#ai'])
    
    def test_zzz_remove_temp_files(self):
        """
        Removes the temporary files that were created for testing.
        The name contains "zzz" because the function are executed in alphabetical order
        and this one should be the last.
        """
        for f in os.listdir():
            if 'temp_test' in f:
                os.remove(f)
        
if __name__ == '__main__':
    unittest.main()