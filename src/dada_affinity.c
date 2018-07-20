/*
 * For binding threads to CPU/Cores
 */

#include "dada_affinity.h"

int dada_bind_thread_to_core(int core)
{
#ifdef HAVE_HWLOC
  int flags = 0;

  // determine the depth of CPU cores in the topology
  int core_depth = hwloc_get_type_or_below_depth (topology, HWLOC_OBJ_CORE);

  int n_cpu_cores = hwloc_get_nbobjs_by_depth (topology, core_depth);

  if (cpu_core >= 0 && cpu_core < n_cpu_cores)
  {
    // get the object corresponding to the specified core
    hwloc_obj_t obj = hwloc_get_obj_by_depth (topology, core_depth, cpu_core);

    // get a copy of its cpuset that we may modify
    hwloc_cpuset_t cpuset = hwloc_bitmap_dup (obj->cpuset);

    // Get only one logical processor (in case the core is SMT/hyper-threaded)
    hwloc_bitmap_singlify(cpuset);

    // try to bind this current process to the CPU set
    if (hwloc_set_cpubind(topology, cpuset, flags))
    {
      char *str;
      int error = errno;
      hwloc_bitmap_asprintf(&str, obj->cpuset);
      cerr << "Could not bind to cpuset " << str << ": " << strerror(error) << endl;
      free(str);
    }

    hwloc_bitmap_free (cpuset);
  }
#else
#ifdef HAVE_AFFINTY
  cpu_set_t set;
  pid_t tpid;

  CPU_ZERO(&set);
  CPU_SET(core, &set);
  tpid = syscall(SYS_gettid);

  if (sched_setaffinity(tpid, sizeof(cpu_set_t), &set) < 0) {
    fprintf(stderr, "failed to set cpu affinity: %s", strerror(errno));
    return -1;
  }

  CPU_ZERO(&set);
  if ( sched_getaffinity(tpid, sizeof(cpu_set_t), &set) < 0 ) {
    fprintf(stderr, "failed to get cpu affinity: %s", strerror(errno));
    return -1;
  }
#endif
#endif
  return 0;
}
