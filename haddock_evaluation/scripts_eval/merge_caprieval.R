library(dplyr)
library(tidyr)
library(readr)

fps = dir(recursive=TRUE, full.names=TRUE, pattern="_result.csv")

erase = function(patt, x) {gsub(patt,"", x)}

all_res = do.call(rbind, lapply(fps, function(fp){
	df = read_csv(fp, show_col_type=FALSE)%>%
		arrange(score)%>%
		mutate(score_rank = order(score))%>%
        mutate(complex=erase("^.+/|_result.csv",fp))
}))

write_tsv(all_res, file="all_caprievals_eval.tsv")

