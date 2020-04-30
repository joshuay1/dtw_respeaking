from praatio import tgio
import os
import shutil
# root = '/home/leferrae/thesis/corpus/grid/gold/'
# for f in os.listdir(root):
#     tg = tgio.openTextgrid(root+f)
#     tier=tg.tierNameList[0]
#     chaine=''
#     for elt in (tg.tierDict[tier].entryList):
#         chaine = chaine+elt[2]+' '
#     with open('/home/leferrae/thesis/injalak/'+f.replace('.TextGrid', '.txt'), mode='w', encoding='utf8') as ficEcr:
#         ficEcr.write(chaine.strip())

root = '/home/getalp/leferrae/thesis/corpora/Kunwinku-speech/forced_align/'

for i in os.listdir(root):
    if 'interview' in i:
        name= i.replace('interview', 'peter_inter_part')
    elif 'CC01-001-BWALKCOMMS1_S9' in i:
        name = i.replace('CC01-001-BWALKCOMMS1_S9', 'joseph_video_part')
    elif 'CC01-001-BWALKCOMMS1_S8' in i:
        name = i.replace('CC01-001-BWALKCOMMS1_S8', 'serena_video_part')
    elif 'CC01-001-BWALKCOMMS1_S10' in i:
        name = i.replace('CC01-001-BWALKCOMMS1_S10', 'mona_video_part')
    elif 'SI1-005-01' in i:
        name = i.replace('SI1-005-01', 'jill_mimih_part')
    elif 'SI1-001-01' in i:
        name = i.replace('SI1-001-01', 'jill_crocs_part')
    else:
        name=i
    shutil.move(root+i, root+name)
    print(name)