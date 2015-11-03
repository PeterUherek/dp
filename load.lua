local t = os.execute("tr '\t' ' '  < sacbee_actions_mapped.tsv | awk '$4==134' | cut -d ' ' -f2 ")
print(t)