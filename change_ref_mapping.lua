function load_tfidf(str)
  local csvFile = io.open("sacbee_tfidf_3-80.csv", 'r')  

  local str = string.gsub(str, "\n", "")
  for line in csvFile:lines('*l') do  

    local l = line:split(';')

    local v = string.find(l[1],str,1,true)
    if v ~= nil then
      csvFile:close()
      print(line)
      return line
    end
  end
  csvFile:close()
end

function load_all_referer()
  local csvFile = io.open("referers_mapping.csv", 'r')
  local newFile = io.open("ref_new.tsv",'w')  

  for line in csvFile:lines('*l') do  
  	local l = line:split(',')
  	print(l[1])
  	local tfifd = load_tfidf(string.format('"%s"',l[1]))
    local str = string.gsub(line, "\n", "")
  	if tfidf==nil then
      print(tfidf)
  		newFile:write(str..",nil\n")
  	else
  		newFile:write(str..","..tfidf.."\n")
  	end
  end
  csvFile:close()
  newFile:close()
end

load_all_referer()
