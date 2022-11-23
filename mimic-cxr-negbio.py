import pandas as pd

df_meta = pd.read_csv('mimic-cxr-2.0.0-metadata.csv')
df_split = pd.read_csv('mimic-cxr-2.0.0-split.csv')

ms = pd.merge(df_meta, df_split, how = 'outer', on = 'dicom_id')       


# test

ms_test

df_negtest = pd.read_csv('mimic-cxr-2.0.0-negbio.csv')
neg_test = pd.merge(ms_test, df_negtest, how='left', left_on='study_id_x', right_on='study_id')
neg_test.drop(['subject_id_x','study_id_x','split'], axis=1)
neg_test = neg_test.reindex(['dicom_id','subject_id', 'study_id', 'No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Opacity', 'Lung Lesion', 
                             'Edema', 'Consolidation', 'Pneumonia', 'Atelectasis', 'Pneumothorax', 'Pleural Effusion', 'Pleural Other', 
                             'Fracture', 'Support Devices', 'ViewPosition', 'ViewCodeSequence_CodeMeaning', 'ProcedureCodeSequence_CodeMeaning'], axis='columns')
neg_test = neg_test.fillna({'No Finding':0, 'Enlarged Cardiomediastinum':0,'Cardiomegaly':0,'Lung Opacity':0,'Lung Lesion':0,
                            'Edema':0,'Consolidation':0,'Pneumonia':0,'Atelectasis':0,'Pneumothorax':0,'Pleural Effusion':0,'Pleural Other':0,
                            'Fracture':0,'Support Devices':0,})


neg_test = neg_test.transpose()
neg_test = neg_test.fillna(method='bfill', axis=0, limit=1)
neg_test = neg_test.transpose()

neg_test.drop(['ViewPosition', 'ProcedureCodeSequence_CodeMeaning'], axis='columns', inplace=True)
neg_test.rename({'ViewCodeSequence_CodeMeaning':'Frontal/Lateral'}, axis=1, inplace=True)
neg_test = neg_test.replace(['CHEST (PORTABLE AP)','postero-anterior','antero-posterior', 'lateral', 'left lateral','CHEST (PA AND LAT)'],
                            ['Frontal','Frontal', 'Frontal', 'Lateral', 'Lateral', ''])

neg_test.to_csv('./mimic-cxr-2.0.0-negbio/test.csv', index=False, mode='w')



# train

ms_train

df_negtrain = pd.read_csv('mimic-cxr-2.0.0-negbio.csv')

neg_train = pd.merge(ms_train, df_negtrain, how='left', left_on='study_id', right_on='study_id')
neg_train.drop(['subject_id_y','split'], axis=1, inplace=True)

neg_train = neg_train.rename({'subject_id_x':'subject_id', 'ViewPosition':'Frontal/Lateral', 'ViewCodeSequence_CodeMeaning':'AP/PA'}, axis=1)
neg_train = neg_train.fillna({'No Finding':0, 'Enlarged Cardiomediastinum':0,'Cardiomegaly':0,'Lung Opacity':0,'Lung Lesion':0,
                              'Edema':0,'Consolidation':0,'Pneumonia':0,'Atelectasis':0,'Pneumothorax':0,'Pleural Effusion':0,'Pleural Other':0,
                              'Fracture':0,'Support Devices':0,})

neg_train = neg_train.transpose()
neg_train = neg_train.fillna(method='bfill', axis=1, limit=1)
neg_train = neg_train.transpose()

neg_train.drop(['ProcedureCodeSequence_CodeMeaning'], axis=1, inplace=True)

neg_train = neg_train.reindex(['dicom_id', 'subject_id', 'study_id', 'Frontal/Lateral', 'AP/PA', 'No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly', 
                               'Lung Opacity', 'Lung Lesion', 'Edema', 'Consolidation', 'Pneumonia', 'Atelectasis', 'Pneumothorax', 'Pleural Effusion', 
                               'Pleural Other', 'Fracture', 'Support Devices'], axis='columns')

neg_train = neg_train.replace(['PA', 'AP', 'LATERAL', 'postero-anterior', 'antero-posterior', 'lateral', 'left lateral'],
                              ['Frontal', 'Frontal', 'Lateral', 'PA', 'AP', '', 'LL'])
neg_train = neg_train.replace({'Frontal/Lateral':'LL'}, 'Lateral')

neg_train.to_csv('train.csv', mode='w', index=False)



#valid

ms_valid

df_negvalid = pd.read_csv('mimic-cxr-2.0.0-negbio.csv')

neg_valid = pd.merge(ms_valid, df_negvalid, how='left', left_on='study_id_x', right_on='study_id')

neg_valid.drop(['subject_id_x','study_id_x','split'], axis=1, inplace=True)
neg_valid = neg_valid.reindex(['dicom_id', 'subject_id', 'study_id', 'ProcedureCodeSequence_CodeMeaning', 'ViewPosition', 'ViewCodeSequence_CodeMeaning', 
                               'No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Opacity', 'Lung Lesion', 'Edema', 'Consolidation', 'Pneumonia', 
                               'Atelectasis', 'Pneumothorax', 'Pleural Effusion', 'Pleural Other', 'Fracture', 'Support Devices'], axis='columns')

neg_valid = neg_valid.fillna({'No Finding':0, 'Enlarged Cardiomediastinum':0,'Cardiomegaly':0,'Lung Opacity':0,'Lung Lesion':0,'Edema':0,'Consolidation':0,'Pneumonia':0,'Atelectasis':0,'Pneumothorax':0,'Pleural Effusion':0,'Pleural Other':0,'Fracture':0,'Support Devices':0,})

neg_valid = neg_valid.replace(['AP', 'PA', 'LATERAL', 'LL', 'antero-posterior', 'postero-anterior', 'left lateral', 'lateral'],
                              ['Frontal','Frontal', 'Lateral', 'Lateral', 'AP', 'PA', 'LL', ''])

neg_valid.rename({'ViewPosition':'Frontal/Lateral', 'ViewCodeSequence_CodeMeaning':'AP/PA'}, axis=1, inplace=True)
neg_valid.drop(['ProcedureCodeSequence_CodeMeaning'], axis=1, inplace=True)

neg_valid.to_csv('valid.csv', mode='w', index=False)
