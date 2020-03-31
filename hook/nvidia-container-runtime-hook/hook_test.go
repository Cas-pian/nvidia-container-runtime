package main

import (
	"fmt"
	"reflect"
	"strings"
	"testing"
)

func TestParseCudaVersionValid(t *testing.T) {
	var tests = []struct {
		version  string
		expected [3]uint32
	}{
		{"0", [3]uint32{0, 0, 0}},
		{"8", [3]uint32{8, 0, 0}},
		{"7.5", [3]uint32{7, 5, 0}},
		{"9.0.116", [3]uint32{9, 0, 116}},
		{"4294967295.4294967295.4294967295", [3]uint32{4294967295, 4294967295, 4294967295}},
	}
	for _, c := range tests {
		vmaj, vmin, vpatch := parseCudaVersion(c.version)
		if vmaj != c.expected[0] || vmin != c.expected[1] || vpatch != c.expected[2] {
			t.Errorf("parseCudaVersion(%s): %d.%d.%d (containerInitInfo: %v)", c.version, vmaj, vmin, vpatch, c.expected)
		}
	}
}

func mustPanic(t *testing.T, f func()) {
	defer func() {
		if err := recover(); err == nil {
			t.Error("Test didn't panic!")
		}
	}()

	f()
}

func TestParseCudaVersionInvalid(t *testing.T) {
	var tests = []string{
		"foo",
		"foo.5.10",
		"9.0.116.50",
		"9.0.116foo",
		"7.foo",
		"9.0.bar",
		"9.4294967296",
		"9.0.116.",
		"9..0",
		"9.",
		".5.10",
		"-9",
		"+9",
		"-9.1.116",
		"-9.-1.-116",
	}
	for _, c := range tests {
		mustPanic(t, func() {
			t.Logf("parseCudaVersion(%s)", c)
			parseCudaVersion(c)
		})
	}
}

func TestGPUUUIDRegexp(t *testing.T) {
	tests := map[string]bool{
		"":                  false,
		"all":               false,
		"none":              false,
		"void":              false,
		"GPU3":              false,
		"GPU-3":             true,
		"gpu-3":             true,
		"gpu-a3g":           false,
		"gpu-a3f":           true,
		"gpu-a3f,":          true,
		"gpu-fa3aa-a,d":     false,
		"GPU-1ef,GPU-2ef":   true,
		"GPU-1ef,GPU-2efx":  false,
		"GPU-a3f,gpu-a3f":   true,
		"GPU-a3f,gpu-a3f,,": true,
		"GPU-a3f-a":         true,
		"GPU-a3f-a,gpu-3af": true,
		"GPU-1ef, ":         false,
		"GPU-83d7ced8-3821-a34c-ce5d-e9264cfa8785":                                          true,
		"GPU-83d7ced8-3821-a34c-ce5d-e9264cfa8785,GPU-83d7ced8-3821-a34c-ce5d-e9264cfa8786": true,
	}

	for str, expected := range tests {
		if match := nvidiaGPUUUIDListExp.MatchString(str); match != expected {
			t.Fatalf("test case failed! %s expected: %v got: %v", str, expected, match)
		} else {
			t.Logf("test case success! %s", str)
		}
	}
}

type containerInitInfo struct {
	startErrStr string // empty means container can be started, otherwise won't started and error message will be set in it.
	*nvidiaConfig
}

type testCase struct {
	Name string
	Envs []string

	ExpectedForOff *containerInitInfo
	ExpectedForOn  *containerInitInfo
}

var nvidiaTestCases = []*testCase{
	{
		Name:           "oridinary_cpu_image",
		Envs:           []string{},
		ExpectedForOff: &containerInitInfo{},
		ExpectedForOn:  &containerInitInfo{},
	}, {
		Name: "old_cuda_image_device_unset_capabilities_unset",
		Envs: []string{"CUDA_VERSION=7.5"},
		ExpectedForOff: &containerInitInfo{
			nvidiaConfig: &nvidiaConfig{
				Devices:      "all",
				Capabilities: allCapabilities,
				Requirements: []string{"cuda>=7.5"},
			},
		},
		ExpectedForOn: &containerInitInfo{
			nvidiaConfig: &nvidiaConfig{
				Capabilities: allCapabilities,
				Requirements: []string{"cuda>=7.5"},
			},
		},
	}, {
		Name: "old_cuda_image_device_all_capabilities_unset",
		Envs: []string{"CUDA_VERSION=7.5", "NVIDIA_VISIBLE_DEVICES=all"},
		ExpectedForOff: &containerInitInfo{
			nvidiaConfig: &nvidiaConfig{
				Devices:      "all",
				Capabilities: allCapabilities,
				Requirements: []string{"cuda>=7.5"},
			},
		},
		ExpectedForOn: &containerInitInfo{
			nvidiaConfig: &nvidiaConfig{
				Capabilities: allCapabilities,
				Requirements: []string{"cuda>=7.5"},
			},
		},
	}, {
		Name: "old_cuda_image_device_id_list_capabilities_unset",
		Envs: []string{"CUDA_VERSION=7.5", "NVIDIA_VISIBLE_DEVICES=0,1"},
		ExpectedForOff: &containerInitInfo{
			nvidiaConfig: &nvidiaConfig{
				Devices:      "0,1",
				Capabilities: allCapabilities,
				Requirements: []string{"cuda>=7.5"},
			},
		},
		ExpectedForOn: &containerInitInfo{
			nvidiaConfig: &nvidiaConfig{
				Devices:      "",
				Capabilities: allCapabilities,
				Requirements: []string{"cuda>=7.5"},
			},
		},
	}, {
		Name: "old_cuda_image_device_uuid_capabilities_unset",
		Envs: []string{"CUDA_VERSION=7.5", "NVIDIA_VISIBLE_DEVICES=GPU-83d7ced8-3821-a34c-ce5d-e9264cfa8785"},
		ExpectedForOff: &containerInitInfo{
			nvidiaConfig: &nvidiaConfig{
				Devices:      "GPU-83d7ced8-3821-a34c-ce5d-e9264cfa8785",
				Capabilities: allCapabilities,
				Requirements: []string{"cuda>=7.5"},
			},
		},
		ExpectedForOn: &containerInitInfo{
			nvidiaConfig: &nvidiaConfig{
				Devices:      "GPU-83d7ced8-3821-a34c-ce5d-e9264cfa8785",
				Capabilities: allCapabilities,
				Requirements: []string{"cuda>=7.5"},
			},
		},
	}, {
		Name:           "old_cuda_image_device_void_capabilities_unset",
		Envs:           []string{"CUDA_VERSION=7.5", "NVIDIA_VISIBLE_DEVICES=void"},
		ExpectedForOff: &containerInitInfo{},
		ExpectedForOn:  &containerInitInfo{},
	}, {
		Name:           "old_cuda_image_device_empty_capabilities_unset",
		Envs:           []string{"CUDA_VERSION=7.5", "NVIDIA_VISIBLE_DEVICES="},
		ExpectedForOff: &containerInitInfo{},
		ExpectedForOn:  &containerInitInfo{},
	}, {
		Name: "old_cuda_image_device_none_capabilities_unset",
		Envs: []string{"CUDA_VERSION=7.5", "NVIDIA_VISIBLE_DEVICES=none"},
		ExpectedForOff: &containerInitInfo{
			nvidiaConfig: &nvidiaConfig{
				Capabilities: allCapabilities,
				Requirements: []string{"cuda>=7.5"},
			},
		},
		ExpectedForOn: &containerInitInfo{
			nvidiaConfig: &nvidiaConfig{
				Capabilities: allCapabilities,
				Requirements: []string{"cuda>=7.5"},
			},
		},
	}, {
		Name: "old_cuda_image_device_none_capabilities_empty",
		Envs: []string{"CUDA_VERSION=7.5", "NVIDIA_VISIBLE_DEVICES=none", "NVIDIA_DRIVER_CAPABILITIES="},
		ExpectedForOff: &containerInitInfo{
			nvidiaConfig: &nvidiaConfig{
				Capabilities: defaultCapability,
				Requirements: []string{"cuda>=7.5"},
			},
		},
		ExpectedForOn: &containerInitInfo{
			nvidiaConfig: &nvidiaConfig{
				Capabilities: defaultCapability,
				Requirements: []string{"cuda>=7.5"},
			},
		},
	}, {
		Name:           "old_cuda_image_device_none_capabilities_set",
		Envs:           []string{"CUDA_VERSION=7.5", "NVIDIA_VISIBLE_DEVICES=", "NVIDIA_DRIVER_CAPABILITIES=graphics,compute,utility"},
		ExpectedForOff: &containerInitInfo{},
		ExpectedForOn:  &containerInitInfo{},
	}, {
		Name:           "new_cuda_image_device_unset_capabilities_unset",
		Envs:           []string{"CUDA_VERSION=9.0.176", "NVIDIA_REQUIRE_CUDA=cuda>=9.0"},
		ExpectedForOff: &containerInitInfo{},
		ExpectedForOn:  &containerInitInfo{},
	}, {
		Name: "new_cuda_image_device_all_capabilities_unset",
		Envs: []string{"CUDA_VERSION=9.0.176", "NVIDIA_REQUIRE_CUDA=cuda>=9.0", "NVIDIA_VISIBLE_DEVICES=all"},
		ExpectedForOff: &containerInitInfo{
			nvidiaConfig: &nvidiaConfig{
				Devices:      "all",
				Capabilities: defaultCapability,
				Requirements: []string{"cuda>=9.0"},
			},
		},
		ExpectedForOn: &containerInitInfo{
			nvidiaConfig: &nvidiaConfig{
				Capabilities: defaultCapability,
				Requirements: []string{"cuda>=9.0"},
			},
		},
	}, {
		Name: "new_cuda_image_device_id_list_capabilities_unset",
		Envs: []string{"CUDA_VERSION=9.0.176", "NVIDIA_REQUIRE_CUDA=cuda>=9.0", "NVIDIA_VISIBLE_DEVICES=0,3"},
		ExpectedForOff: &containerInitInfo{
			nvidiaConfig: &nvidiaConfig{
				Devices:      "0,3",
				Capabilities: defaultCapability,
				Requirements: []string{"cuda>=9.0"},
			},
		},
		ExpectedForOn: &containerInitInfo{
			nvidiaConfig: &nvidiaConfig{
				Capabilities: defaultCapability,
				Requirements: []string{"cuda>=9.0"},
			},
		},
	}, {
		Name: "new_cuda_image_device_uuid_capabilities_unset",
		Envs: []string{"CUDA_VERSION=9.0.176", "NVIDIA_REQUIRE_CUDA=cuda>=9.0", "NVIDIA_VISIBLE_DEVICES=GPU-83d7ced8-3821-a34c-ce5d-e9264cfa8785"},
		ExpectedForOff: &containerInitInfo{
			nvidiaConfig: &nvidiaConfig{
				Devices:      "GPU-83d7ced8-3821-a34c-ce5d-e9264cfa8785",
				Capabilities: defaultCapability,
				Requirements: []string{"cuda>=9.0"},
			},
		},
		ExpectedForOn: &containerInitInfo{
			nvidiaConfig: &nvidiaConfig{
				Devices:      "GPU-83d7ced8-3821-a34c-ce5d-e9264cfa8785",
				Capabilities: defaultCapability,
				Requirements: []string{"cuda>=9.0"},
			},
		},
	}, {
		Name:           "new_cuda_image_device_void_capabilities_unset",
		Envs:           []string{"CUDA_VERSION=9.0.176", "NVIDIA_REQUIRE_CUDA=cuda>=9.0", "NVIDIA_VISIBLE_DEVICES=void"},
		ExpectedForOff: &containerInitInfo{},
		ExpectedForOn:  &containerInitInfo{},
	}, {
		Name:           "new_cuda_image_device_empty_capabilities_unset",
		Envs:           []string{"CUDA_VERSION=9.0.176", "NVIDIA_REQUIRE_CUDA=cuda>=9.0", "NVIDIA_VISIBLE_DEVICES="},
		ExpectedForOff: &containerInitInfo{},
		ExpectedForOn:  &containerInitInfo{},
	}, {
		Name: "new_cuda_image_device_none_capabilities_unset",
		Envs: []string{"CUDA_VERSION=9.0.176", "NVIDIA_REQUIRE_CUDA=cuda>=9.0", "NVIDIA_VISIBLE_DEVICES=none"},
		ExpectedForOff: &containerInitInfo{
			nvidiaConfig: &nvidiaConfig{
				Capabilities: defaultCapability,
				Requirements: []string{"cuda>=9.0"},
			},
		},
		ExpectedForOn: &containerInitInfo{
			nvidiaConfig: &nvidiaConfig{
				Capabilities: defaultCapability,
				Requirements: []string{"cuda>=9.0"},
			},
		},
	}, {
		Name: "new_cuda_image_device_none_capabilities_empty",
		Envs: []string{"CUDA_VERSION=9.0.176", "NVIDIA_REQUIRE_CUDA=cuda>=9.0", "NVIDIA_VISIBLE_DEVICES=none"},
		ExpectedForOff: &containerInitInfo{
			nvidiaConfig: &nvidiaConfig{
				Capabilities: defaultCapability,
				Requirements: []string{"cuda>=9.0"},
			},
		},
		ExpectedForOn: &containerInitInfo{
			nvidiaConfig: &nvidiaConfig{
				Capabilities: defaultCapability,
				Requirements: []string{"cuda>=9.0"},
			},
		},
	}, {
		Name: "new_cuda_image_device_none_capabilities_set",
		Envs: []string{"CUDA_VERSION=9.0.176", "NVIDIA_REQUIRE_CUDA=cuda>=9.0", "NVIDIA_VISIBLE_DEVICES=none", "NVIDIA_DRIVER_CAPABILITIES=graphics,compute,utility"},
		ExpectedForOff: &containerInitInfo{
			nvidiaConfig: &nvidiaConfig{
				Capabilities: "graphics,compute,utility",
				Requirements: []string{"cuda>=9.0"},
			},
		},
		ExpectedForOn: &containerInitInfo{
			nvidiaConfig: &nvidiaConfig{
				Capabilities: "graphics,compute,utility",
				Requirements: []string{"cuda>=9.0"},
			},
		},
	}, {
		Name: "new_cuda_image_multi_device_env_capabilities_empty_0",
		Envs: []string{"CUDA_VERSION=9.0.176", "NVIDIA_REQUIRE_CUDA=cuda>=9.0", "NVIDIA_VISIBLE_DEVICES=none", "NVIDIA_VISIBLE_DEVICES=gpu-1"},
		ExpectedForOff: &containerInitInfo{
			nvidiaConfig: &nvidiaConfig{
				Devices:      "gpu-1",
				Capabilities: defaultCapability,
				Requirements: []string{"cuda>=9.0"},
			},
		},
		ExpectedForOn: &containerInitInfo{
			nvidiaConfig: &nvidiaConfig{
				Devices:      "gpu-1",
				Capabilities: defaultCapability,
				Requirements: []string{"cuda>=9.0"},
			},
		},
	}, {
		Name: "new_cuda_image_multi_device_env_capabilities_empty_1",
		Envs: []string{"CUDA_VERSION=9.0.176", "NVIDIA_REQUIRE_CUDA=cuda>=9.0", "NVIDIA_VISIBLE_DEVICES=none", "NVIDIA_VISIBLE_DEVICES=0,1"},
		ExpectedForOff: &containerInitInfo{
			nvidiaConfig: &nvidiaConfig{
				Devices:      "0,1",
				Capabilities: defaultCapability,
				Requirements: []string{"cuda>=9.0"},
			},
		},
		ExpectedForOn: &containerInitInfo{
			nvidiaConfig: &nvidiaConfig{
				Capabilities: defaultCapability,
				Requirements: []string{"cuda>=9.0"},
			},
		},
	}, {
		Name: "new_cuda_image_multi_device_env_capabilities_empty_2",
		Envs: []string{"CUDA_VERSION=9.0.176", "NVIDIA_REQUIRE_CUDA=cuda>=9.0", "NVIDIA_VISIBLE_DEVICES=", "NVIDIA_VISIBLE_DEVICES=0,3"},
		ExpectedForOff: &containerInitInfo{
			nvidiaConfig: &nvidiaConfig{
				Devices:      "0,3",
				Capabilities: defaultCapability,
				Requirements: []string{"cuda>=9.0"},
			},
		},
		ExpectedForOn: &containerInitInfo{},
	}, {
		Name: "new_cuda_image_multi_device_env_capabilities_empty_3",
		Envs: []string{"CUDA_VERSION=9.0.176", "NVIDIA_REQUIRE_CUDA=cuda>=9.0", "NVIDIA_VISIBLE_DEVICES=none", "NVIDIA_VISIBLE_DEVICES=GPU-3", "NVIDIA_VISIBLE_DEVICES=0,2"},
		ExpectedForOff: &containerInitInfo{
			nvidiaConfig: &nvidiaConfig{
				Devices:      "0,2",
				Capabilities: defaultCapability,
				Requirements: []string{"cuda>=9.0"},
			},
		},
		ExpectedForOn: &containerInitInfo{
			nvidiaConfig: &nvidiaConfig{
				Devices:      "GPU-3",
				Capabilities: defaultCapability,
				Requirements: []string{"cuda>=9.0"},
			},
		},
	},
}

func TestSwitchOfMountByUUID(t *testing.T) {

	doHook := func(t *testCase, hook *HookConfig) (nvidiaConfig *nvidiaConfig, e error) {
		defer func() {
			if err := recover(); err != nil {
				if e1, ok := err.(string); ok {
					e = fmt.Errorf("%s", e1)
				}
			}
		}()

		s := &Spec{
			Process: &Process{Env: t.Envs},
		}

		env := getEnvMap(s.Process.Env, hook.MountGPUOnlyByUUID)
		envSwarmGPU = hook.SwarmResource
		nvidiaConfig = getNvidiaConfig(env, hook.MountGPUOnlyByUUID)
		return nvidiaConfig, e
	}

	runTest := func(mountGPUOnlyByUUID bool, c *testCase, cii *containerInitInfo) {
		hook := &HookConfig{MountGPUOnlyByUUID: mountGPUOnlyByUUID}
		n, err := doHook(c, hook)
		if err == nil {
			if cii.startErrStr == "" && reflect.DeepEqual(n, cii.nvidiaConfig) {
				t.Logf("testcase success! mountGPUOnlyByUUID=%v %s\t start success!", mountGPUOnlyByUUID, c.Name)
			} else {
				t.Logf("expected nvidiaConfig= %#v\n\t\t   got nvidiaConfig= %#v\n ", cii.nvidiaConfig, n)
				t.Fatalf("testcase failed! mountGPUOnlyByUUID=%v %s\t start but wrong info %v %v %v", mountGPUOnlyByUUID, c.Name, cii.startErrStr, n == nil, cii.nvidiaConfig == nil)
			}
		} else if err != nil {
			if cii.startErrStr != "" && strings.TrimSuffix(err.Error(), "\n") == cii.startErrStr {
				t.Logf("testcase success! mountGPUOnlyByUUID=%v %s\t won't start!", mountGPUOnlyByUUID, c.Name)
			} else {
				t.Logf("expected err= %#v\n\t\t   got err= %#v\n ", err.Error(), cii.startErrStr)
				t.Fatalf("testcase failed! mountGPUOnlyByUUID=%v %s\t won't start!", mountGPUOnlyByUUID, c.Name)
			}
		}
	}

	for _, c := range nvidiaTestCases {
		runTest(false, c, c.ExpectedForOff)
		runTest(true, c, c.ExpectedForOn)
	}
}
