package main

import (
	"encoding/json"
	"fmt"
	"log"
	"os"
	"path"
	"regexp"
	"strconv"
	"strings"
)

var envSwarmGPU *string

const (
	envNVRequirePrefix      = "NVIDIA_REQUIRE_"
	envLegacyCUDAVersion    = "CUDA_VERSION"
	envNVRequireCUDA        = envNVRequirePrefix + "CUDA"
	envNVGPU                = "NVIDIA_VISIBLE_DEVICES"
	envNVDriverCapabilities = "NVIDIA_DRIVER_CAPABILITIES"
	defaultCapability       = "utility"
	allCapabilities         = "compute,compat32,graphics,utility,video,display"
	envNVDisableRequire     = "NVIDIA_DISABLE_REQUIRE"

	// Please referer to these docs:
	// https://docs.nvidia.com/deploy/nvml-api/group__nvmlDeviceQueries.html#group__nvmlDeviceQueries_1g84dca2d06974131ccec1651428596191
	// https://github.com/NVIDIA/libnvidia-container/blob/master/src/cli/common.c#L11
	// If GPU UUID is wrong or doesn't exist, nvidia-container-cli which is called by this hook will report with failure
	nvidiaGPUUUIDListFmt = `^[gG][pP][uU]-([0-9a-fA-F-]){1,75}(,|,[gG][pP][uU]-([0-9a-fA-F-]){1,75})*$`

	errGPUCanOnlyBeUsedByUUID = "Wrong way to use GPUs! " +
		"If you dont't need GPU, use an image without CUDA, or build images with env " + envNVGPU + "=none. " +
		"Otherwise set pod.spec.containers[*].resources.requests['nvidia.com/gpu'] for kubernetes, " +
		"or set env " + envNVGPU + "={GPU UUID} for docker. "
)

var nvidiaGPUUUIDListExp = regexp.MustCompile(nvidiaGPUUUIDListFmt)
var noneGPU = "none"

type nvidiaConfig struct {
	Devices        string
	Capabilities   string
	Requirements   []string
	DisableRequire bool
}

type containerConfig struct {
	Pid    int
	Rootfs string
	Env    map[string]string
	Nvidia *nvidiaConfig
}

// github.com/opencontainers/runtime-spec/blob/v1.0.0/specs-go/config.go#L94-L100
type Root struct {
	Path string `json:"path"`
}

// github.com/opencontainers/runtime-spec/blob/v1.0.0/specs-go/config.go#L30-L57
type Process struct {
	Env []string `json:"env,omitempty"`
}

// We use pointers to structs, similarly to the latest version of runtime-spec:
// https://github.com/opencontainers/runtime-spec/blob/v1.0.0/specs-go/config.go#L5-L28
type Spec struct {
	Process *Process `json:"process,omitempty"`
	Root    *Root    `json:"root,omitempty"`
}

type HookState struct {
	Pid int `json:"pid,omitempty"`
	// After 17.06, runc is using the runtime spec:
	// github.com/docker/runc/blob/17.06/libcontainer/configs/config.go#L262-L263
	// github.com/opencontainers/runtime-spec/blob/v1.0.0/specs-go/state.go#L3-L17
	Bundle string `json:"bundle"`
	// Before 17.06, runc used a custom struct that didn't conform to the spec:
	// github.com/docker/runc/blob/17.03.x/libcontainer/configs/config.go#L245-L252
	BundlePath string `json:"bundlePath"`
}

func parseCudaVersion(cudaVersion string) (vmaj, vmin, vpatch uint32) {
	if _, err := fmt.Sscanf(cudaVersion, "%d.%d.%d\n", &vmaj, &vmin, &vpatch); err != nil {
		vpatch = 0
		if _, err := fmt.Sscanf(cudaVersion, "%d.%d\n", &vmaj, &vmin); err != nil {
			vmin = 0
			if _, err := fmt.Sscanf(cudaVersion, "%d\n", &vmaj); err != nil {
				log.Panicln("invalid CUDA version:", cudaVersion)
			}
		}
	}

	return
}

func getEnvMap(e []string, mountGPUOnlyByUUID bool) (m map[string]string) {
	m = make(map[string]string)
	for _, s := range e {
		p := strings.SplitN(s, "=", 2)
		if len(p) != 2 {
			log.Panicln("environment error")
		}

		if mountGPUOnlyByUUID && p[0] == envNVGPU {
			if _, in := m[p[0]]; !in || (in && nvidiaGPUUUIDListExp.MatchString(p[1])) {
				// the last value with 'GPU-' prefix has the highest priority, otherwise use the first value
				m[p[0]] = p[1]
			}
		} else {
			m[p[0]] = p[1]
		}
	}
	return
}

func loadSpec(path string) (spec *Spec) {
	f, err := os.Open(path)
	if err != nil {
		log.Panicln("could not open OCI spec:", err)
	}
	defer f.Close()

	if err = json.NewDecoder(f).Decode(&spec); err != nil {
		log.Panicln("could not decode OCI spec:", err)
	}
	if spec.Process == nil {
		log.Panicln("Process is empty in OCI spec")
	}
	if spec.Root == nil {
		log.Panicln("Root is empty in OCI spec")
	}
	return
}

func getDevices(env map[string]string, mountGPUOnlyByUUID bool) *string {
	gpuVars := []string{envNVGPU}
	if envSwarmGPU != nil {
		// The Swarm resource has higher precedence.
		gpuVars = append([]string{*envSwarmGPU}, gpuVars...)
	}

	var ret *string
	for _, gpuVar := range gpuVars {
		if devices, ok := env[gpuVar]; ok {
			ret = &devices
		}
	}

	if !mountGPUOnlyByUUID { // old way
		return ret
	}

	if ret == nil || *ret == "" || *ret == "void" || *ret == "none" {
		// handle empty, 'void', 'none' first, cause different logic between old and new CUDA images
		// new cuda image: unset and empty equals void
		// old cuda image: unset means all, empty equals void
		return ret
	}

	// disable use GPU on value: all or 0,1,2,3, only GPU UUID list seperated by ',' is supported,
	// so that in k8s no GPU will be mounted in multi containers (allocated by scheduler and set by device plugin)
	if nvidiaGPUUUIDListExp.MatchString(*ret) {
		return ret
	}

	log.Println(errGPUCanOnlyBeUsedByUUID)
	return &noneGPU // should not execute this
}

func getCapabilities(env map[string]string) *string {
	if capabilities, ok := env[envNVDriverCapabilities]; ok {
		return &capabilities
	}
	return nil
}

func getRequirements(env map[string]string) []string {
	// All variables with the "NVIDIA_REQUIRE_" prefix are passed to nvidia-container-cli
	var requirements []string
	for name, value := range env {
		if strings.HasPrefix(name, envNVRequirePrefix) {
			requirements = append(requirements, value)
		}
	}
	return requirements
}

// Mimic the new CUDA images if no capabilities or devices are specified.
func getNvidiaConfigLegacy(env map[string]string, mountGPUOnlyByUUID bool) *nvidiaConfig {
	var devices string
	if d := getDevices(env, mountGPUOnlyByUUID); d == nil {
		if !mountGPUOnlyByUUID {
			// Environment variable unset: default to "all".
			devices = "all"
		} else {
			devices = "none"
			log.Println(errGPUCanOnlyBeUsedByUUID)
		}
	} else if len(*d) == 0 || *d == "void" {
		// Environment variable empty or "void": not a GPU container.
		return nil
	} else {
		// Environment variable non-empty and not "void".
		devices = *d
	}
	if devices == "none" {
		devices = ""
	}

	var capabilities string
	if c := getCapabilities(env); c == nil {
		// Environment variable unset: default to "all".
		capabilities = allCapabilities
	} else if len(*c) == 0 {
		// Environment variable empty: use default capability.
		capabilities = defaultCapability
	} else {
		// Environment variable non-empty.
		capabilities = *c
	}
	if capabilities == "all" {
		capabilities = allCapabilities
	}

	requirements := getRequirements(env)

	vmaj, vmin, _ := parseCudaVersion(env[envLegacyCUDAVersion])
	cudaRequire := fmt.Sprintf("cuda>=%d.%d", vmaj, vmin)
	requirements = append(requirements, cudaRequire)

	// Don't fail on invalid values.
	disableRequire, _ := strconv.ParseBool(env[envNVDisableRequire])

	return &nvidiaConfig{
		Devices:        devices,
		Capabilities:   capabilities,
		Requirements:   requirements,
		DisableRequire: disableRequire,
	}
}

func getNvidiaConfig(env map[string]string, mountGPUOnlyByUUID bool) *nvidiaConfig {
	legacyCudaVersion := env[envLegacyCUDAVersion]
	cudaRequire := env[envNVRequireCUDA]
	if len(legacyCudaVersion) > 0 && len(cudaRequire) == 0 {
		// Legacy CUDA image detected.
		return getNvidiaConfigLegacy(env, mountGPUOnlyByUUID)
	}

	var devices string
	if d := getDevices(env, mountGPUOnlyByUUID); d == nil || len(*d) == 0 || *d == "void" {
		// Environment variable unset or empty or "void": not a GPU container.
		return nil
	} else {
		// Environment variable non-empty and not "void".
		devices = *d
	}
	if devices == "none" {
		devices = ""
	}

	var capabilities string
	if c := getCapabilities(env); c == nil || len(*c) == 0 {
		// Environment variable unset or set but empty: use default capability.
		capabilities = defaultCapability
	} else {
		// Environment variable set and non-empty.
		capabilities = *c
	}
	if capabilities == "all" {
		capabilities = allCapabilities
	}

	requirements := getRequirements(env)

	// Don't fail on invalid values.
	disableRequire, _ := strconv.ParseBool(env[envNVDisableRequire])

	return &nvidiaConfig{
		Devices:        devices,
		Capabilities:   capabilities,
		Requirements:   requirements,
		DisableRequire: disableRequire,
	}
}

func getContainerConfig(hook HookConfig) (config containerConfig) {
	var h HookState
	d := json.NewDecoder(os.Stdin)
	if err := d.Decode(&h); err != nil {
		log.Panicln("could not decode container state:", err)
	}

	b := h.Bundle
	if len(b) == 0 {
		b = h.BundlePath
	}

	s := loadSpec(path.Join(b, "config.json"))

	env := getEnvMap(s.Process.Env, hook.MountGPUOnlyByUUID)
	envSwarmGPU = hook.SwarmResource
	return containerConfig{
		Pid:    h.Pid,
		Rootfs: s.Root.Path,
		Env:    env,
		Nvidia: getNvidiaConfig(env, hook.MountGPUOnlyByUUID),
	}
}
