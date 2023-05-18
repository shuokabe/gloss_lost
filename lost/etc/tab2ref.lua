#!/usr/bin/env lua

local st = 0
local toks, poss, lbl = {}, {}
local max = 0
local function finish()
	if not lbl then
		return
	end
	max = math.max(max, #toks)
	local tok = table.concat(toks, "_")
	local pos = table.concat(poss, "_")
	print(st, st + 1, tok.."|"..pos, lbl)
	st, toks, poss, lbl = st + 1, {}, {}
end
for line in io.lines() do
	if line == "" then
		finish()
		print(st)
		print("EOS")
		st = 0
	else
		local tok, pos, chk = line:match("(%S+)%s+(%S+)%s+(%S+)")
		local knd = chk:sub(1, 1)
		if knd ~= "O" then
			chk = chk:sub(3, -1)
		end
		if knd == "O" then
			finish()
			print(st, st + 1, tok.."|"..pos, chk)
			st = st + 1
		elseif knd == "B" then
			finish()
			toks[#toks + 1] = tok
			poss[#poss + 1] = pos
			lbl = chk
		elseif knd == "I" then
			toks[#toks + 1] = tok
			poss[#poss + 1] = pos
		end
	end
end

