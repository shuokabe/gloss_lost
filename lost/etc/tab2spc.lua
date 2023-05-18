#!/usr/bin/env lua

local mlen = tonumber(({...})[1] or 4)

local alls = {"O", "ADJP", "ADVP", "CONJP", "INTJ", "NP", "PP", "PRT", "SBAR", "VP"}
local lbls = {"ADJP", "ADVP", "CONJP", "INTJ", "NP", "PP", "PRT", "SBAR", "VP"}

local function out(from, to, obs, lbl)
	local base = from.."\t"..to.."\t"..obs
	for _, l in ipairs(lbl) do
		print(base, l)
	end
end
local function gen(toks, poss, at, len)
	local tok = table.concat(toks, "_", at, at + len - 1)
	local pos = table.concat(poss, "_", at, at + len - 1)
	out(at - 1, at - 1 + len, tok.."|"..pos, lbls)
end

local st = 0
local toks, poss = {}, {}
for line in io.lines() do
	if line == "" then
		for len = 2, mlen do
			for pos = 1, #toks - len + 1 do
				gen(toks, poss, pos, len)
			end
		end
		print(st)
		print("EOS")
		st, toks, poss = 0, {}, {}
	else
		local tok, pos = line:match("(%S+)%s+(%S+)")
		out(st, st + 1, tok.."|"..pos, alls)
		toks[#toks + 1] = tok
		poss[#poss + 1] = pos
		st = st + 1
	end
end
os.exit(0)


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

