import os
import pandas as pd


class ReadExtraData(object):
    def read_data(self):
        bureau_data = pd.read_csv(os.path.join('Data','bureau.csv'))
        bureau_bal_data = pd.read_csv(os.path.join('Data', 'bureau_balance.csv'))
        prev_application_data = pd.read_csv(os.path.join('Data','previous_application.csv'))
        pos_cash_bal_data = pd.read_csv(os.path.join('Data','POS_CASH_balance.csv'))
        instalments_pay_data = pd.read_csv(os.path.join('Data', 'installments_payments.csv'))
        credit_card_bal_data = pd.read_csv(os.path.join('Data', 'credit_card_balance.csv'))
        return bureau_data, bureau_bal_data, prev_application_data, pos_cash_bal_data,\
               instalments_pay_data, credit_card_bal_data




