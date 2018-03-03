from difflib import SequenceMatcher
import copy

def similar(a, b):
    return SequenceMatcher(None, a, b).ratio()

print similar('apple','Apple'.lower())

team_spellings  = list()
scoring_offense = dict()

ts_file = open('WDataFiles/WTeamSpellings.csv')
so_file = open('WDataFiles/WScoringOffense.csv')
    
for i, line in enumerate(so_file):
    if i == 0:
        continue
    team_name_so = line.split(',')[2]
    if not scoring_offense.has_key(team_name_so):
        scoring_offense[team_name_so] = dict()
        
for i, line in enumerate(ts_file):
    if i == 0:
        continue
    team_name_ts = line.split(',')[0]
    team_num_ts  = int(line.split(',')[1])
    team_spellings.append((team_name_ts,team_num_ts))
    
ts_file.close()
so_file.close()
    
i = 0
for team in scoring_offense:
    i += 1
    print str(i) + '/' + str(len(scoring_offense))
    highest_similarity = 0
    for item in team_spellings:
        current_team_name  = item[0]
        current_similarity = similar(team.lower(), current_team_name)
        if current_similarity > highest_similarity:
            matched_team_name = current_team_name
            matched_ID        = item[1]
            #print matched_team_name
            highest_similarity = current_similarity*1.    
    scoring_offense[team]['Name']   = matched_team_name
    scoring_offense[team]['TeamID'] = matched_ID
    a = 0 
    
so_output_file = open('WDataFiles/WScoringOffense_IDs.csv','w')
so_file        = open('WDataFiles/WScoringOffense.csv')

for i, line in enumerate(so_file):
    if i == 0:
        new_line = line[:-2] + ',Kaggle Name,Kaggle TeamID' + line[-2:]
        so_output_file.write(new_line)
    else:
        team_name_so = line.split(',')[2]
        new_line = line[:-2] + ',' + scoring_offense[team_name_so]['Name'] + ',' + str(scoring_offense[team_name_so]['TeamID']) + line[-2:]
        so_output_file.write(new_line)
    
so_output_file.close()
so_file.close()
    
pass
            

