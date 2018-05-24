from Scripts.EncodeCategoricalFeaturesInDF import EncodeCategoricalFeaturesInDF
from Scripts.PreprocessApplicationData import PreprocessApplicationData
from Scripts.ReadExtraData import ReadExtraData

preprocess_application_data = False
if preprocess_application_data:
    preprocess_application_data_object = PreprocessApplicationData()
    application_data = preprocess_application_data_object.return_application_data()

    encode_categorical_features_in_df_object = EncodeCategoricalFeaturesInDF()
    application_data_encoded = encode_categorical_features_in_df_object.encode_features(application_data)


read_extra_data_object = ReadExtraData()
bureau_data, bureau_bal_data, prev_application_data, pos_cash_bal_data,\
               instalments_pay_data, credit_card_bal_data = read_extra_data_object.read_data()
