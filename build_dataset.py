import pandas as pd
import shutil


folders_path='data/dogsandcats/'
csv_path='data/sampleSubmission.csv'
id='id'
label='label'
image_fomrat='.jpg'
df = pd.read_csv (csv_path)
print (df[id].iloc[0])

data_lenght=len(df[id])
for i in range(data_lenght):
    if  df[label].iloc[i]!=0:
        file_name=str(df[id].iloc[i])+image_fomrat

        #Path(folders_path+'dog/'+file_name).rename(folders_path+'test/'+file_name)
        from_path=folders_path+'dog/'+file_name
        to_path=folders_path+'cat/'+file_name
        try:
            print('fil', file_name ,'moved to',to_path)
            shutil.move(from_path ,to_path)

        except:pass
   # df[id].iloc[0]