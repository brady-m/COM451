
int arg_index = 0;
char *arg_option;
char *pvcon = NULL;

char crack(int argc, char** argv, char* flags, int ignore_unknowns)
{
    char *pv, *flgp;

    while ((arg_index) < argc){
        if (pvcon != NULL)
            pv = pvcon;
        else{
            if (++arg_index >= argc) return(0);
            pv = argv[arg_index];
            if (*pv != '-')
                return(0);
            }
        pv++;

        if (*pv != 0){
            if ((flgp=strchr(flags,*pv)) != NULL){
                pvcon = pv;
                if (*(flgp+1) == '|') { arg_option = pv+1; pvcon = NULL; }
                return(*pv);
                }
            else
                if (!ignore_unknowns){
                    fprintf(stderr, "%s: no such flag: %s\n", argv[0], pv);
                    return(EOF);
                    }
                else pvcon = NULL;
	    	}
        pvcon = NULL;
    }

    return(0);
}
