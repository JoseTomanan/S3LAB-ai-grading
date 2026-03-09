<script lang="ts">
  let logs: string[] = $state([]);

  const original = {
    log: console.log.bind(console),
    warn: console.warn.bind(console),
    error: console.error.bind(console),
  };

  console.log = (...args) => { logs.push(args.map(String).join(' ')); original.log(...args); };
  console.warn = (...args) => { logs.push('⚠️ ' + args.map(String).join(' ')); original.warn(...args); };
  console.error = (...args) => { logs.push('🔴 ' + args.map(String).join(' ')); original.error(...args); };
</script>


<div class="fixed bottom-0 left-0 right-0 z-50 max-h-48 overflow-y-auto
            bg-black/80 text-green-400 text-xs font-mono p-2 space-y-0.5">
  {#each logs as log}
    <div>{log}</div>
  {/each}
</div>