#!/usr/bin/env lua
for line in io.lines() do
	for itm in line:gmatch("%S+") do
		local wrd, pos, chk = itm:match("(.+)|(.+)@(.+)")
		if chk == "O" then
			print(wrd, chk)
		else
			local lbl = "B-"..chk
			for tok in wrd:gmatch("[^_]+") do
				print(tok, lbl)
				lbl = "I-"..chk
			end
		end
	end
	print()
end

