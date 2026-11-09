# d16_summarise.R -- summaries of the D16 sweep used by fig:thdisc_price_cycles (thesis Discussion,
# Section 6.4): the slope of ln(long-cycle amplitude) on the 1961 secondary share for every pair of
# lead times (Panel a) and the band-passed ln price at three shares (Panel c).
# Run from final_model/ after d16_cycles.R:  Rscript discussion/d16_summarise.R
OUT <- file.path("discussion", "outputs")
res <- read.csv(file.path(OUT, "d16_sweep.csv"))
m <- res[res$set == "main", ]
rows <- list()
for (key in unique(paste(m$tau_E, m$tau_R))) {
  parts <- strsplit(key, " ")[[1]]; tE <- as.numeric(parts[1]); tR <- as.numeric(parts[2])
  g <- m[m$tau_E == tE & m$tau_R == tR & m$solved, ]
  if (nrow(g) >= 3) {
    b  <- coef(lm(log(amp_long) ~ sigma0, data = g))[2]
    bE <- coef(lm(log(sd_dlnE)  ~ sigma0, data = g))[2]
    rows[[length(rows) + 1]] <- data.frame(tau_E = tE, tau_R = tR, n_solved = nrow(g),
      sig_max = max(g$sigma0), slope_lnamp = unname(b), slope_lnsdE = unname(bE),
      amp_ratio = g$amp_long[nrow(g)] / g$amp_long[1])
  } else {
    rows[[length(rows) + 1]] <- data.frame(tau_E = tE, tau_R = tR, n_solved = nrow(g),
      sig_max = if (nrow(g)) max(g$sigma0) else NA, slope_lnamp = NA, slope_lnsdE = NA, amp_ratio = NA)
  }
}
write.csv(do.call(rbind, rows), file.path(OUT, "d16_direction_by_delay.csv"), row.names = FALSE)
# band-passed ln price, run_all.R band() of Chapter 3: loess spans 15.5 and 55.8 yr of the 62-yr window
tr <- read.csv(file.path(OUT, "d16_trajectories.csv"))
lo_y <- function(x, yrs) { N <- length(x); as.numeric(predict(loess(x ~ seq_len(N), span = min(yrs / N, 1), degree = 1))) }
long_b <- function(x) lo_y(x, 15.5) - lo_y(x, 55.8)
h <- subset(tr, forcing == "historical")
out <- do.call(rbind, lapply(split(h, h$sigma0), function(d) { d <- d[order(d$year), ]
  data.frame(sigma0 = d$sigma0, year = d$year, band = long_b(log(d$P))) }))
write.csv(out, file.path(OUT, "d16_band_historical.csv"), row.names = FALSE)
