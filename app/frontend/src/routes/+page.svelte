<script lang="ts">
  import logo from '$lib/assets/android-chrome-512x512.png';
  import * as Sheet from '$lib/components/ui/sheet/index.ts';
  import Manual from './Manual.svelte';
  import { getApiStatus, type ApiStatus } from '$lib/utils/api.ts';
  import type { PageData } from './$types.ts';
  import IconInstances from '~icons/mdi/file-document-multiple-outline';
  import IconSections from '~icons/mdi/account-group-outline';
  import IconDone from '~icons/mdi/check-circle-outline';
  import IconPending from '~icons/mdi/progress-clock';
  import IconChevronRight from '~icons/mdi/chevron-right';
  import IconManual from '~icons/mdi/help-circle-outline';

  let { data }: { data: PageData } = $props();

  let apiStatus = $state<ApiStatus | 'checking'>('checking');
  let manualOpen = $state(false);

  $effect.pre(() => {
    let cancelled = false;
    (async () => {
      const status = await getApiStatus();
      if (!cancelled) apiStatus = status;
    })();
    return () => { cancelled = true };
  });

  const statusColor = $derived(
    apiStatus === 'online' ? 'bg-secondary-700' :
    apiStatus === 'checking' ? 'bg-muted-foreground' :
    'bg-destructive'
  );
  const statusLabel = $derived(
    apiStatus === 'online' ? 'Backend online' :
    apiStatus === 'checking' ? 'Connecting…' :
    apiStatus === 'no-ai-key' ? 'No AI key' : 'Offline'
  );
  const statusTextColor = $derived(
    apiStatus === 'online' ? 'text-secondary-700' :
    apiStatus === 'checking' ? 'text-muted-foreground' :
    'text-destructive'
  );
  const statusBgColor = $derived(
    apiStatus === 'online' ? 'bg-secondary-300' :
    apiStatus === 'checking' ? 'bg-muted' :
    'bg-destructive/10'
  );
</script>

<!-- Hero -->
<div class="bg-card border-b border-border px-6 pt-10 pb-8 flex flex-col items-center gap-4">
  <!-- help button -->
  <button
    onclick={() => manualOpen = true}
    class="absolute top-4 right-4 size-9 rounded-full bg-background border border-border flex items-center justify-center text-muted-foreground hover:text-foreground hover:bg-border transition-colors"
  >
    <IconManual class="size-5" />
  </button>

  <!-- logo ring -->
  <div class="size-20 rounded-full bg-accent border border-border flex items-center justify-center">
    <img src={logo} alt="SIPAT.MATH logo" class="size-[52px] rounded-full object-contain" />
  </div>

  <!-- wordmark -->
  <div class="text-center -mt-1">
    <div
      class="font-heading font-black text-[38px] leading-[0.92] tracking-[-0.06em]"
    >
      <span class="text-foreground">SIPAT</span><br />
      <span class="text-primary-700">MATH</span>
    </div>
    <p class="text-[12px] text-foreground/55 uppercase tracking-[0.12em] mt-1.5 opacity-100">AI Grading System</p>
  </div>

  <!-- status chip -->
  <div class="flex items-center gap-1.5 {statusBgColor} rounded-full px-2.5 py-1">
    <span class="size-[7px] rounded-full {statusColor}"></span>
    <span class="text-[11px] font-semibold {statusTextColor}">{statusLabel}</span>
  </div>
</div>

<!-- Nav cards + recent -->
<div class="px-4 py-5 flex flex-col gap-3">
  <!-- 2-column nav grid -->
  <div class="grid grid-cols-2 gap-2.5">
    <a
      href="/instances"
      class="bg-card rounded-xl shadow-sm px-3.5 py-[18px] flex flex-col gap-2.5 no-underline hover:-translate-y-0.5 hover:shadow-md transition-all active:translate-y-0"
    >
      <div class="size-10 rounded-[10px] bg-accent flex items-center justify-center">
        <IconInstances class="size-[22px] text-foreground" />
      </div>
      <div>
        <p class="font-heading text-[14px] font-extrabold text-foreground tracking-[-0.03em] opacity-100">Test Instances</p>
        <p class="text-[11px] text-foreground/55 leading-[1.4] mt-0.5 opacity-100">Manage exams &amp; grading</p>
      </div>
    </a>

    <a
      href="/sections"
      class="bg-card rounded-xl shadow-sm px-3.5 py-[18px] flex flex-col gap-2.5 no-underline hover:-translate-y-0.5 hover:shadow-md transition-all active:translate-y-0"
    >
      <div class="size-10 rounded-[10px] bg-accent flex items-center justify-center">
        <IconSections class="size-[22px] text-foreground" />
      </div>
      <div>
        <p class="font-heading text-[14px] font-extrabold text-foreground tracking-[-0.03em] opacity-100">Sections</p>
        <p class="text-[11px] text-foreground/55 leading-[1.4] mt-0.5 opacity-100">Manage class list</p>
      </div>
    </a>
  </div>

  <!-- Recent -->
  {#if data.recent.length > 0}
    <div>
      <p class="font-heading text-[13px] font-bold text-foreground/55 uppercase tracking-[0.04em] mb-2">Recent</p>
      <div class="flex flex-col gap-2">
        {#each data.recent as inst}
          <a
            href="/instances/{inst.test_id}/papers"
            class="bg-card rounded-xl shadow-sm px-4 py-3 flex items-center gap-3 no-underline hover:shadow-md transition-shadow"
          >
            <div class="size-[38px] rounded-[10px] flex items-center justify-center shrink-0
              {inst.is_done_rendering ? 'bg-secondary-300' : 'bg-accent'}">
              {#if inst.is_done_rendering}
                <IconDone class="size-5 text-secondary-700" />
              {:else}
                <IconPending class="size-5 text-primary-700" />
              {/if}
            </div>
            <div class="flex-1 min-w-0">
              <p class="text-[14px] font-medium text-foreground truncate opacity-100">{inst.name}</p>
              <p class="text-[11px] text-foreground/55 mt-0.5 opacity-100">
                {new Date(inst.date).toLocaleDateString()}
              </p>
            </div>
            <IconChevronRight class="size-5 text-foreground/55 shrink-0" />
          </a>
        {/each}
      </div>
    </div>
  {/if}
</div>

<!-- Manual help sheet -->
<Sheet.Root bind:open={manualOpen}>
  <Sheet.Content side="bottom" class="bg-card rounded-t-2xl max-h-[80dvh] overflow-y-auto">
    <Manual />
  </Sheet.Content>
</Sheet.Root>
