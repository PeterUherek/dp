require 'math'

--local file1 = io.open("sacbee_day/number_days_form_creation_date.txt", 'w') 
--local file2 = io.open("sacbee_day/number_of_hours_from_first_access.txt", 'w')

function get_minutes_and_day(str)
  local str = str:split('T')
  local date = str[1]:split('-')
  local time = str[2]:split(':')

  local minutes = (tonumber(time[1])*60)+tonumber(time[2])
  local week_day = get_day_of_week(tonumber(date[3]),tonumber(date[2]),2015)
  
  local month_day = tonumber(date[3])

  return minutes,week_day,month_day
end

function get_day_of_week(dd, mm, yy)
  dw=os.date('*t',os.time{year=yy,month=mm,day=dd})['wday']
  return dw,({"Sun","Mon","Tue","Wed","Thu","Fri","Sat" })[dw]
end


function get_sin(minutes,all)
	return math.sin(math.pi*minutes/all) --sin(pi*hour/maxTime)
end

function get_cos(minutes,all)
	return math.cos(math.pi*minutes/all)
end

function please_get_time(str)
  
  local minutes,week_day,month_day = get_minutes_and_day(str)
  local output = {}

  output[1] = get_sin(minutes,1440)
  output[2] = get_cos(minutes,1440)
  output[3] = get_sin(week_day,7)
  output[4] = get_cos(week_day,7)
  output[5] = get_sin(month_day,31)
  output[6] = get_cos(month_day,31)

  return output
end

function is_weekend(str)
  
  local minutes,week_day,month_day = get_minutes_and_day(str)
  local weekend = 0

  if week_day == 1 or week_day == 7 then
    weekend = 1
  end
  
  return weekend
end

function get_week_day(str)
  local minutes,week_day,month_day = get_minutes_and_day(str)
  return weekend
end

--- [[FUNCTION FOR TASK 3]] ---
function get_time_for_date_of_acess(str)
  str = format_string(str)
  local date = str:split('-')
  local week_day = get_day_of_week(tonumber(date[3]),tonumber(date[2]),2015)
  local month_day = tonumber(date[3])

  local output = {}
  output[1] = get_sin(week_day,7)
  output[2] = get_cos(week_day,7)
  output[3] = get_sin(month_day,31)
  output[4] = get_cos(month_day,31)

  return output
end

function get_time_for_time_of_acess(str)
  local output = {}
  output[1] = get_sin(tonumber(str),24)
  output[2] = get_cos(tonumber(str),24)

  return output
end

function get_time_in_days(creation_date,current_date)
  local crea = creation_date:split('-')
  local curr = current_date:split('-')

  date1 = os.time{year=tonumber(crea[1]), month=tonumber(crea[2]), day=tonumber(crea[3])}
  date2 = os.time{year=2015, month=tonumber(curr[2]), day=tonumber(curr[3])}
  diff = math.abs(os.difftime(date1, date2) / (24 * 60 * 60)) 

  --file1:write(tostring(diff)..","..tostring(crea[1])..","..tostring(crea[2])..","..tostring(crea[3])..","..tostring(curr[2])..","..tostring(curr[3]).."\n")
  return diff
end

function get_time_in_hours(first_date,current_date,first_time,current_time)
  local firs = first_date:split('-')
  local curr = current_date:split('-')
  
  date1 = os.time{year=2015, month=tonumber(firs[2]), day=tonumber(firs[3]), hour = tonumber(first_time)}
  date2 = os.time{year=2015, month=tonumber(curr[2]), day=tonumber(curr[3]), hour = tonumber(current_time)}
  diff = math.abs(os.difftime(date1, date2) / (60 * 60)) 

  --file2:write(tostring(diff)..","..tostring(crea[2])..","..tostring(crea[3])..","..tostring(crea[3])..""..",".."\n")
  return diff
end

