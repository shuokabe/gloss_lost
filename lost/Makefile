CFLAGS=-std=c99 -W -Wall -Wextra -O3
LIBS  =-lm -lpthread

DESTDIR=
PREFIX =/usr/local

INSTALL= install -p
INSTALL_EXEC= $(INSTALL) -m 0755
INSTALL_DATA= $(INSTALL) -m 0644

lost: src/lost.c
	@echo "[CC] src/lost.c --> lost"
	@$(CC) -DNDEBUG $(CFLAGS) -o lost src/lost.c $(LIBS)

debug: src/lost.c
	@echo "[CC] src/lost.c --> lost"
	@$(CC) -g $(CFLAGS) -o lost src/lost.c

install: lost
	@echo "[CP] lost --> $(DESTDIR)$(PREFIX)/bin"
	@mkdir -p $(DESTDIR)$(PREFIX)/bin
	@$(INSTALL_EXEC) lost $(DESTDIR)$(PREFIX)/bin

clean:
	@echo "[RM] lost"
	@rm -f lost

.PHONY: clean install

