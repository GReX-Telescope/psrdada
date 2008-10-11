
#define IBOB_VLAN_BASE "169.254.128."
#define IBOB_PORT 23

static const char emulate_telnet_msg1[] = { 255, 251, 37,
				            255, 253, 38,
				            255, 251, 38,
				            255, 253, 3,
				            255, 251, 24,
				            255, 251, 31,
				            255, 251, 32,
				            255, 251, 33,
				            255, 251, 34,
				            255, 251, 39,
				            255, 253, 5,
				            255, 251, 35,
				            0 };

static const char emulate_telnet_resp[] = { 255, 251, 1, 
                                            255, 251, 3, 
                                            0 };

static const char emulate_telnet_msg2[] = { 255, 253, 1, 0 };

static const char ibob_prompt[] = "IBOB % ";

