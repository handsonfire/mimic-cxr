
import pandas as pd

# metadata와 split 기준이 되는 내용을 dicom_id 기준으로 병합

df_meta = pd.read_csv('mimic-cxr-2.0.0-metadata.csv')
df_split = pd.read_csv('mimic-cxr-2.0.0-split.csv')
ms = pd.merge(df_meta, df_split, how = 'outer', on = 'dicom_id')  


# 병합한 데이터에서 'test' 데이터들만 분류해고, 필요 없는 데이터들 정리

ms_test = ms[ms['split'] == 'test']
ms_test.rename({'subject_id_x':'subject_id', 'study_id_x':'study_id'}, axis=1)
ms_test = ms_test.drop(['PerformedProcedureStepDescription','Rows', 'Columns', 'StudyDate', 'StudyTime', 'study_id_y', 'subject_id_y',
                        'PatientOrientationCodeSequence_CodeMeaning'], axis=1)


# 정리된 데이터프레임에 chexpert 파일의 데이터를, study_id 기준으로 병합

df_chextest = pd.read_csv('mimic-cxr-2.0.0-chexpert.csv')
test_df = pd.merge(ms_test, df_chextest, how='left', left_on='study_id_x', right_on='study_id')

# 중복된 내용 지우기, 인덱스 순서 정리하기, 결측값 표시를 0으로 바꾸기 등 chexpert 파일과 비슷하게 수정

test_df.drop(['subject_id_x','study_id_x','split'], axis=1, inplace=True)
test_df = test_df.reindex(['dicom_id','subject_id', 'study_id', 'No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Opacity', 'Lung Lesion',
                           'Edema', 'Consolidation', 'Pneumonia', 'Atelectasis', 'Pneumothorax', 'Pleural Effusion', 'Pleural Other', 'Fracture', 
                           'Support Devices', 'ViewCodeSequence_CodeMeaning', 'ViewPosition', 'ProcedureCodeSequence_CodeMeaning'], axis='columns')
test_df = test_df.fillna({'No Finding':0, 'Enlarged Cardiomediastinum':0,'Cardiomegaly':0,'Lung Opacity':0,
                          'Lung Lesion':0,'Edema':0,'Consolidation':0,'Pneumonia':0,'Atelectasis':0,'Pneumothorax':0,'Pleural Effusion':0,
                          'Pleural Other':0,'Fracture':0,'Support Devices':0,})


# "ViewPosition" 인덱스 중 결측값에, 같은 행의"ProcedureCodeSequence_CodeMeaning"의 정보 입력하기

test_df = test_df.transpose()
test_df = test_df.fillna(method='bfill', axis=0, limit=1)
test_df = test_df.transpose()


# 입력된 정보를 'Frontal/Lateral'로 바꿔주기 
test_df = test_df.replace(['CHEST (PORTABLE AP)','AP','PA', 'LATERAL', 'LL', 'CHEST (PA AND LAT)'],
                          ['Frontal','Frontal', 'Frontal', 'Lateral', 'Lateral', ''])
test_df.rename({'ViewPosition':'Frontal/Lateral'}, axis=1, inplace=True)
test_df.drop(['ViewCodeSequence_CodeMeaning','ProcedureCodeSequence_CodeMeaning'], axis=1, inplace=True)


# 파일로 저장
test_df.to_csv('test.csv', mode='w', index=False)



# train도 비슷한 순서로 정리

ms_train = ms[ms['split']=='train']
ms_train = ms_train.rename({'subject_id_x':'subject_id', 'study_id_x':'study_id'}, axis=1)
ms_train.drop(['PerformedProcedureStepDescription','Rows', 'Columns', 'StudyDate', 'StudyTime', 'study_id_y', 'subject_id_y',
               'PatientOrientationCodeSequence_CodeMeaning'], axis=1, inplace = True)

df_chextrain = pd.read_csv('mimic-cxr-2.0.0-chexpert.csv')

train_df = pd.merge(ms_train, df_chextrain, how='left', left_on='study_id', right_on='study_id')

train_df.drop(['subject_id_y','split'], axis=1, inplace=True)
train_df = train_df.rename({'subject_id_x':'subject_id', 'ViewPosition':'Frontal/Lateral', 'ViewCodeSequence_CodeMeaning':'AP/PA'}, axis=1)
train_df = train_df.fillna({'No Finding':0, 'Enlarged Cardiomediastinum':0,'Cardiomegaly':0,'Lung Opacity':0,'Lung Lesion':0,'Edema':0,
                            'Consolidation':0,'Pneumonia':0,'Atelectasis':0,'Pneumothorax':0,'Pleural Effusion':0,'Pleural Other':0,'Fracture':0,'Support Devices':0,})

train_df = train_df.transpose()
train_df = train_df.fillna(method='bfill', axis=1, limit=1)
train_df = train_df.transpose()

train_df.drop(['ProcedureCodeSequence_CodeMeaning'], axis=1, inplace=True)

train_df = train_df.reindex(['dicom_id', 'subject_id', 'study_id', 'Frontal/Lateral', 'AP/PA', 'No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly', 
                             'Lung Opacity', 'Lung Lesion', 'Edema', 'Consolidation', 'Pneumonia', 'Atelectasis', 'Pneumothorax', 
                             'Pleural Effusion', 'Pleural Other', 'Fracture', 'Support Devices'], axis='columns')

train_df = train_df.replace(['PA', 'AP', 'LATERAL', 'postero-anterior', 'antero-posterior', 'lateral', 'left lateral'],
                            ['Frontal', 'Frontal', 'Lateral', 'PA', 'AP', '', 'LL'])
train_df = train_df.replace({'Frontal/Lateral':'LL'}, 'Lateral')

train_df.to_csv('train.csv', mode='w', index=False)


# valid

ms_valid = ms[ms['split']=='validate']
ms_valid.rename({'subject_id_x':'subject_id', 'study_id_x':'study_id'}, axis=1)
ms_valid = ms_valid.drop(['PerformedProcedureStepDescription','Rows', 'Columns', 'StudyDate', 'StudyTime', 'study_id_y', 'subject_id_y', 
                          'PatientOrientationCodeSequence_CodeMeaning'], axis=1)


df_chexvalid = pd.read_csv('mimic-cxr-2.0.0-chexpert.csv')

valid_df = pd.merge(ms_valid, df_chextest, how='left', left_on='study_id_x', right_on='study_id')

valid_df.drop(['subject_id_x','study_id_x','split'], axis=1, inplace=True)
valid_df = valid_df.reindex(['dicom_id', 'subject_id', 'study_id', 'ProcedureCodeSequence_CodeMeaning', 'ViewPosition', 'ViewCodeSequence_CodeMeaning', 
                             'No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Opacity', 'Lung Lesion', 'Edema', 'Consolidation', 'Pneumonia',
                             'Atelectasis', 'Pneumothorax', 'Pleural Effusion', 'Pleural Other', 'Fracture', 'Support Devices'], axis='columns')

valid_df = valid_df.fillna({'No Finding':0, 'Enlarged Cardiomediastinum':0,'Cardiomegaly':0,'Lung Opacity':0,'Lung Lesion':0,
                            'Edema':0,'Consolidation':0,'Pneumonia':0,'Atelectasis':0,'Pneumothorax':0,'Pleural Effusion':0,'Pleural Other':0,
                            'Fracture':0,'Support Devices':0,})

valid_df = valid_df.replace(['AP', 'PA', 'LATERAL', 'LL', 'antero-posterior', 'postero-anterior', 'left lateral', 'lateral'],
                            ['Frontal','Frontal', 'Lateral', 'Lateral', 'AP', 'PA', 'LL', ''])
valid_df.rename({'ViewPosition':'Frontal/Lateral', 'ViewCodeSequence_CodeMeaning':'AP/PA'}, axis=1, inplace=True)
valid_df.drop(['ProcedureCodeSequence_CodeMeaning'], axis=1, inplace=True)

valid_df.to_csv('valid.csv', mode='w', index=False)
