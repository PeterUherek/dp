require 'math'

function get_minutes(str)
  local str = str:split('T')
  local date = str[1]:split('-')
  local time = str[2]:split(':')

  local minutes = (tonumber(time[1])*60)+tonumber(time[2])
  return minutes
end

function get_sin(minutes)
	return math.sin(2*math.pi*minutes/1440) --sin(2*pi*hour/maxTime)
end

function get_cos(minutes)
	return math.cos(2*math.pi*minutes/1440)
end

function please_get_sin_and_cos(str)
  
  local minutes = get_minutes(str)

  local sin = get_sin(minutes)
  local cos = get_cos(minutes)
  
  return sin,cos
end

--please_get_me_time("0015-03-01T21:11:50+00:00")