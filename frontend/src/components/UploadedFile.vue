<script setup>
import { ref } from "vue";
import UploadedFileModal from "@/components/UploadedFileModal.vue";

const props = defineProps({
  file: {
    type: Object,
    required: true,
  },
  formatFileSize: {
    type: Function,
    required: true,
  },
});

const emit = defineEmits(["remove"]);

const isModalOpen = ref(false);

const openModal = () => {
  isModalOpen.value = true;
};

const closeModal = () => {
  isModalOpen.value = false;
};

const onRemove = () => {
  emit("remove", props.file.id);
};
</script>

<template>
  <div
    class="flex items-center p-2 border border-stone-500 rounded-xl max-w-xs"
  >
    <svg
      class="size-5 fill-pink-500/50 mr-2 shrink-0"
      viewBox="0 0 24 24"
      fill="none"
      xmlns="http://www.w3.org/2000/svg"
    >
      <g id="SVGRepo_bgCarrier" stroke-width="0"></g>
      <g
        id="SVGRepo_tracerCarrier"
        stroke-linecap="round"
        stroke-linejoin="round"
      ></g>
      <g id="SVGRepo_iconCarrier">
        <path
          d="M15.7997 2.21048C15.3897 1.80048 14.6797 2.08048 14.6797 2.65048V6.14048C14.6797 7.60048 15.9197 8.81048 17.4297 8.81048C18.3797 8.82048 19.6997 8.82048 20.8297 8.82048C21.3997 8.82048 21.6997 8.15048 21.2997 7.75048C19.8597 6.30048 17.2797 3.69048 15.7997 2.21048Z"
        ></path>
        <path
          d="M20.5 10.19H17.61C15.24 10.19 13.31 8.26 13.31 5.89V3C13.31 2.45 12.86 2 12.31 2H8.07C4.99 2 2.5 4 2.5 7.57V16.43C2.5 20 4.99 22 8.07 22H15.93C19.01 22 21.5 20 21.5 16.43V11.19C21.5 10.64 21.05 10.19 20.5 10.19ZM11.5 17.75H7.5C7.09 17.75 6.75 17.41 6.75 17C6.75 16.59 7.09 16.25 7.5 16.25H11.5C11.91 16.25 12.25 16.59 12.25 17C12.25 17.41 11.91 17.75 11.5 17.75ZM13.5 13.75H7.5C7.09 13.75 6.75 13.41 6.75 13C6.75 12.59 7.09 12.25 7.5 12.25H13.5C13.91 12.25 14.25 12.59 14.25 13C14.25 13.41 13.91 13.75 13.5 13.75Z"
        ></path>
      </g>
    </svg>
    <div class="flex flex-col grow min-w-0">
      <span
        class="text-sm font-medium truncate block text-stone-400 hover:underline cursor-pointer"
        :title="props.file.name + ' 파일 보기'"
        @click="openModal"
      >
        {{ props.file.name }}
      </span>
      <span class="text-xs text-stone-500">
        {{ props.formatFileSize(props.file.size) }}
      </span>
    </div>

    <button
      @click="onRemove"
      class="group ml-2 p-2 rounded-full hover:bg-stone-700/50 flex shrink-0 cursor-pointer"
      :aria-label="'Remove file ' + props.file.name"
      title="업로드 취소"
    >
      <svg
        class="size-4 fill-stone-500 group-hover:fill-stone-400 transition duration-200 ease-in-out"
        fill="none"
        viewBox="0 0 24 24"
        stroke="currentColor"
      >
        <path
          stroke-linecap="round"
          stroke-linejoin="round"
          stroke-width="2"
          d="M6 18L18 6M6 6l12 12"
        />
      </svg>
    </button>
  </div>

  <UploadedFileModal
    :is-open="isModalOpen"
    :file="props.file"
    @close="closeModal"
  />
</template>