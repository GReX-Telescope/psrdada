
# scripts distributed as is (installed in share/)
data_DATA = apsr.cfg

dist_data_DATA = apsr.info \
                 apsr.viewer \
                 dspsr.cal \
                 dspsr.default \
                 dspsr.multi \
                 dspsr.select \
                 dspsr.single \
                 dspsr.singleF \
                 singleF.psh \
                 apsr.scratch \
                 multi.txt \
                 pdfb3_1024_1024_2048 \
                 pdfb3_1024_256_2048 \
                 pdfb3_2048_512_2048 \
                 pdfb3_256_256_2048 \
                 pdfb3_512_256_2048 \
                 pdfb3_512_512_2048 \
                 names_J0737-3039.txt \
                 names_J0024-7204.txt \
                 names_J1910-5959.txt \
                 names_J1701-3006.txt \
                 names_J1748-2446.txt


nobase_dist_data_DATA = testing/CAL.spec \
                        testing/J0437_1405MHz.spec \
                        testing/J0437_2500MHz_dbdisk.spec \
                        testing/J0437_2500MHz.spec \
                        testing/J0737-3038.spec


edit = sed \
	-e 's|@datadir[@]|$(pkgdatadir)|g' \
	-e 's|@prefix[@]|$(prefix)|g'

dspsr.singleF: Makefile
	rm -f $@ $@.tmp
	$(edit) '$(srcdir)/$@.in' >$@.tmp
	chmod +x $@.tmp
	chmod a-w $@.tmp
	mv $@.tmp $@

dspsr.singleF: $(srcdir)/dspsr.singleF.in

DIFFDATA=$(nobase_dist_data_DATA) $(dist_data_DATA) $(data_DATA)

diff:
	@for file in $(DIFFDATA); do \
	 if diff $$file $(datadir)/$$file > /dev/null; then \
	  echo > /dev/null ; \
	 else \
	  echo "###########################################################"; \
	  echo "#"; \
	  echo "#"; \
	  echo diff $$file $(datadir)/$$file; \
	  diff $$file $(datadir)/$$file; \
	  echo "#"; \
	  echo "#"; \
	 fi; \
	done

