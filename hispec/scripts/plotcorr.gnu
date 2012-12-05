set autoscale y
set key inside left top vertical Right noreverse enhanced autotitles box linetype -1 linewidth 1.000
plot "bob4.LACSPC" binary format="%float" array=512 with linespoint

set term png
set output "output.png"
replot

unset key
