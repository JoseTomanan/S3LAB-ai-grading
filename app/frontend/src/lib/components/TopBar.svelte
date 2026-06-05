<script lang="ts">
  import type { Snippet } from 'svelte';
  import IconChevronLeft from '~icons/mdi/chevron-left';

  interface Props {
    title: string;
    sub?: string;
    back?: string;
    right?: Snippet;
  }

  let { title, sub, back, right }: Props = $props();
</script>

<header class="bg-card border-b border-border flex items-center h-14 px-3.5 gap-2 shrink-0">
  {#if back}
    <a
      href={back}
      class="size-9 rounded-full bg-background border border-border flex items-center justify-center text-muted-foreground hover:text-foreground hover:bg-border transition-colors shrink-0"
    >
      <IconChevronLeft class="size-5" />
    </a>
  {:else}
    <div class="w-9 shrink-0"></div>
  {/if}

  <div class="flex-1 text-center min-w-0">
    <div class="font-heading text-[16px] font-extrabold text-foreground tracking-[-0.04em] leading-tight truncate">
      {title}
    </div>
    {#if sub}
      <div class="text-[11px] text-foreground/55 mt-0.5 font-mono truncate">{sub}</div>
    {/if}
  </div>

  {#if right}
    <div class="flex gap-1.5 items-center shrink-0">
      {@render right()}
    </div>
  {:else}
    <div class="w-9 shrink-0"></div>
  {/if}
</header>
