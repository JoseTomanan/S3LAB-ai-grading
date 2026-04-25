<script lang="ts">
  import type { Snippet } from 'svelte';
  import * as Sheet from '$lib/components/ui/sheet/index.ts';

  interface Props {
    open?: boolean;
    title: string;
    children: Snippet;
  }

  let { open = $bindable(false), title, children }: Props = $props();
</script>

<Sheet.Root bind:open>
  <Sheet.Content
    side="bottom"
    class="bg-card rounded-t-2xl border-t border-border px-5 pb-8 pt-6 gap-0 max-w-[480px] mx-auto [&>button:last-child]:hidden"
  >
    <div class="w-9 h-1 rounded-full bg-foreground/20 mx-auto mb-5"></div>

    <Sheet.Header class="p-0 mb-5 text-left space-y-0">
      <Sheet.Title class="font-heading text-[18px] font-extrabold tracking-[-0.04em] text-foreground">
        {title}
      </Sheet.Title>
    </Sheet.Header>

    {@render children()}
  </Sheet.Content>
</Sheet.Root>
