<script lang="ts">
  const { data, children } = $props();
  import IconBack from "~icons/mdi/arrow-left";
</script>


<nav class="bg-sidebar text-sidebar-foreground
            shadow shadow-sidebar-border
            w-full flex flex-row justify-between p-4 pt-6">
  <span class="*:leading-5">
    <h1 class="align-left italic">Class List</h1>
    <h6>SECTION_ID {data.section_id}</h6>
  </span>
  <a href="/sections">
    <IconBack class="size-8"/>
  </a>
</nav>

<span class="-my-4"></span>

{@render children()}

