struct AParams {
  bool  verbose;
  int   runMode;
//  int   myParam1;
  bool  isMultithread;
};
int usage();
int setDefaults(AParams *PARAMS);
int viewParams(const AParams *PARAMS);
void showDeviceInformation();