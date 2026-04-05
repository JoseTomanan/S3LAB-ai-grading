<script lang="ts">
  import { navigating } from '$app/state';
  const { data, children } = $props();
  
  import { Button } from '$lib/components/CustomButton/index.ts';
  import { Separator } from '$lib/components/ui/separator';
  import { Skeleton } from '$lib/components/ui/skeleton';
  import IconBack from "~icons/mdi/chevron-left";
  import IconSectionList from '~icons/mdi/format-list-bulleted';

  const sectionId = $derived(Number(data.section_id));
  const sectionName = $derived(data.sections.find(
    (section) => section.section_id === sectionId
  )!.section_name);
</script>



<nav class="bg-sidebar text-sidebar-foreground
            shadow shadow-sidebar-border
            w-full flex flex-col gap-y-2.5 p-4 pt-6">
  <div class="w-full inline-flex items-center justify-between">
    <Button variant="floating" href="/sections">
      <IconBack class="size-8"/>
    </Button>
    <span class="*:leading-5 *:text-center">
      <h2>{sectionName}</h2>
    </span>
    <Button variant="floating" href="/sections">
      <IconSectionList class="size-7 m-0.5"/>
    </Button>
  </div>

  <Separator/>
</nav>

<span class="-my-4"></span>

{#if navigating.to}
  <div class="container">
    <Skeleton class="grayscale-50 w-full h-64 rounded-none"/>
  </div>

{:else}
  {@render children()}
{/if}
